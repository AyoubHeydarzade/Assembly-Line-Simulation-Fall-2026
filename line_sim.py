"""line_sim.py — Discrete-event simulator for a two-station manual assembly line
with per-bin load cells.

Model
-----
Two stations in series with one finite buffer between them. Each station has a
row of parts bins. Building one unit at a station means working through those
bins in order:

    pick from bin 1 -> work (drawn duration) -> pick from bin 2 -> work -> ...

The first pick at bin 1 is what starts the unit. Every pick removes
`parts_per_pick` parts from that bin, so the load cell under it sees the weight
drop. Parts remaining are inferred the way a real load cell would infer them:

    parts_remaining = weight_g / part_weight_g

Step durations are lognormal with the mean and coefficient of variation given
per bin, so a station's cycle time is the sum of its own step durations.

The engine is event-driven throughout. A station that cannot hand off its
finished unit enters a `blocked` state and is released by the downstream pull
rather than by polling. Busy / blocked / starved time is accumulated on every
state transition, which is what the utilization and line-balance KPIs are
computed from.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

WORKING = "working"
BLOCKED = "blocked"
STARVED = "starved"
STALLED = "stalled"  # bin empty and auto-refill is off


# ---------------------------------------------------------------------------
# Specifications (what the dashboard edits)
# ---------------------------------------------------------------------------
@dataclass
class BinSpec:
    name: str
    start_weight_g: float   # load-cell reading with the bin full
    part_weight_g: float    # mass of one part
    parts_per_pick: int     # parts removed in one pick
    mean_step_s: float      # mean duration of the step this pick begins
    cv: float = 0.2         # coefficient of variation of that duration


@dataclass
class StationSpec:
    name: str
    bins: List[BinSpec]

    @property
    def nominal_cycle_s(self) -> float:
        """Sum of mean step times — the cycle time with no variability."""
        return float(sum(b.mean_step_s for b in self.bins))


@dataclass
class BufferSpec:
    capacity: int


# ---------------------------------------------------------------------------
# Engine internals
# ---------------------------------------------------------------------------
@dataclass(order=True)
class Event:
    time: float
    seq: int
    kind: str
    payload: Any = field(compare=False, default=None)


@dataclass
class _StationRT:
    """Mutable runtime state for one station."""
    spec: StationSpec
    weights: List[float]
    picks: List[int]
    refills: List[int]
    state: str = STARVED
    since: float = 0.0
    item: Optional[int] = None
    step: int = 0
    item_started_t: float = 0.0
    held_item: Optional[int] = None   # finished unit held while blocked
    busy_s: float = 0.0
    blocked_s: float = 0.0
    starved_s: float = 0.0


class TwoStationLine:
    def __init__(
        self,
        stations: List[StationSpec],
        buffer_spec: BufferSpec,
        sim_time_s: float = 3600.0,
        seed: int | None = 42,
        auto_refill: bool = True,
    ):
        assert len(stations) == 2, "this model is two stations"
        self.sim_time_s = float(sim_time_s)
        self.auto_refill = auto_refill
        self.rng = np.random.default_rng(seed)

        self.stations = [
            _StationRT(
                spec=s,
                weights=[b.start_weight_g for b in s.bins],
                picks=[0] * len(s.bins),
                refills=[0] * len(s.bins),
            )
            for s in stations
        ]
        self.buffer_cap = int(buffer_spec.capacity)
        self.buffer: List[int] = []

        self.now = 0.0
        self._seq = 0
        self.event_q: List[Event] = []

        self.item_counter = 0
        self.item_start_t: Dict[int, float] = {}
        self.events: List[Dict[str, Any]] = []
        self.readings: List[Dict[str, Any]] = []
        self.station_cycles: List[Dict[str, Any]] = []
        self.completed: List[Dict[str, Any]] = []
        self.buffer_trace: List[Dict[str, Any]] = [{"t": 0.0, "buffer_len": 0}]

    # -- plumbing ----------------------------------------------------------
    def _schedule(self, delay: float, kind: str, payload: Any = None) -> None:
        self._seq += 1
        heapq.heappush(self.event_q, Event(self.now + delay, self._seq, kind, payload))

    def _draw(self, mean: float, cv: float) -> float:
        """Lognormal with exactly this mean and coefficient of variation."""
        if cv <= 0 or mean <= 0:
            return float(mean)
        sigma2 = math.log(1.0 + cv * cv)
        mu = math.log(mean) - 0.5 * sigma2
        return float(self.rng.lognormal(mu, math.sqrt(sigma2)))

    def _accrue(self, rt: _StationRT, t: float) -> None:
        dt = max(0.0, t - rt.since)
        if rt.state == WORKING:
            rt.busy_s += dt
        elif rt.state == BLOCKED:
            rt.blocked_s += dt
        else:  # STARVED or STALLED
            rt.starved_s += dt
        rt.since = t

    def _set_state(self, rt: _StationRT, new_state: str) -> None:
        self._accrue(rt, self.now)
        rt.state = new_state

    def _log(self, event: str, si: int, bi: Optional[int], item_id: Optional[int], **extra) -> None:
        rt = self.stations[si]
        rec: Dict[str, Any] = {
            "t": self.now,
            "event": event,
            "station": si,
            "station_name": rt.spec.name,
            "bin": bi,
            "bin_name": rt.spec.bins[bi].name if bi is not None else None,
            "item_id": item_id,
        }
        rec.update(extra)
        self.events.append(rec)
        if bi is not None and "weight_g" in extra:
            pw = rt.spec.bins[bi].part_weight_g
            self.readings.append(
                {**rec, "parts_est": (extra["weight_g"] / pw) if pw > 0 else float("nan")}
            )

    # -- load cells --------------------------------------------------------
    def _pick(self, si: int, bi: int, item_id: int) -> bool:
        """Remove one pick worth of parts from a bin. False = bin is empty."""
        rt = self.stations[si]
        b = rt.spec.bins[bi]
        need = b.parts_per_pick * b.part_weight_g

        if rt.weights[bi] + 1e-9 < need:
            self._log("stockout", si, bi, item_id, weight_g=rt.weights[bi])
            if not self.auto_refill:
                return False
            rt.weights[bi] = b.start_weight_g
            rt.refills[bi] += 1
            self._log("refill", si, bi, item_id, weight_g=rt.weights[bi])

        rt.weights[bi] -= need
        rt.picks[bi] += 1
        self._log(
            "pick", si, bi, item_id,
            weight_g=rt.weights[bi], removed_g=need, parts_taken=b.parts_per_pick,
        )
        return True

    # -- flow --------------------------------------------------------------
    def _begin_item(self, si: int, item_id: int) -> None:
        rt = self.stations[si]
        rt.item = item_id
        rt.step = 0
        rt.item_started_t = self.now
        self._set_state(rt, WORKING)
        self._log("station_start", si, None, item_id)
        self._begin_step(si)

    def _begin_step(self, si: int) -> None:
        rt = self.stations[si]
        bi = rt.step
        if not self._pick(si, bi, rt.item):
            self._set_state(rt, STALLED)
            return
        dur = self._draw(rt.spec.bins[bi].mean_step_s, rt.spec.bins[bi].cv)
        self._schedule(dur, "step_done", (si, bi, rt.item))

    def _on_step_done(self, si: int, bi: int, item_id: int) -> None:
        rt = self.stations[si]
        self._log("step_done", si, bi, item_id, weight_g=rt.weights[bi])
        rt.step += 1
        if rt.step < len(rt.spec.bins):
            self._begin_step(si)
        else:
            self._finish_station(si, item_id)

    def _finish_station(self, si: int, item_id: int) -> None:
        rt = self.stations[si]
        self._log("station_finish", si, None, item_id)
        self.station_cycles.append({
            "station": si,
            "station_name": rt.spec.name,
            "item_id": item_id,
            "start_t": rt.item_started_t,
            "finish_t": self.now,
            "cycle_s": self.now - rt.item_started_t,
        })
        rt.item = None

        if si == len(self.stations) - 1:
            self.completed.append({
                "item_id": item_id,
                "t": self.now,
                "lead_s": self.now - self.item_start_t.get(item_id, self.now),
            })
            self._log("complete", si, None, item_id)
            self._set_state(rt, STARVED)
            self._try_start(si)
            return

        if len(self.buffer) < self.buffer_cap:
            self._push_buffer(item_id)
            self._set_state(rt, STARVED)
            self._try_start(si)
            self._try_start(si + 1)
        else:
            rt.held_item = item_id
            self._set_state(rt, BLOCKED)

    def _push_buffer(self, item_id: int) -> None:
        self.buffer.append(item_id)
        self._log("to_buffer", 0, None, item_id, buffer_len=len(self.buffer))
        self.buffer_trace.append({"t": self.now, "buffer_len": len(self.buffer)})

    def _release_blocked(self) -> None:
        """Downstream freed a slot — hand off whatever upstream is holding."""
        up = self.stations[0]
        if up.state == BLOCKED and up.held_item is not None and len(self.buffer) < self.buffer_cap:
            item_id = up.held_item
            up.held_item = None
            self._push_buffer(item_id)
            self._set_state(up, STARVED)
            self._try_start(0)

    def _try_start(self, si: int) -> None:
        rt = self.stations[si]
        if rt.item is not None or rt.state in (BLOCKED, STALLED):
            return
        if si == 0:
            self.item_counter += 1
            item_id = self.item_counter
            self.item_start_t[item_id] = self.now
            self._begin_item(0, item_id)
        elif self.buffer:
            item_id = self.buffer.pop(0)
            self._log("from_buffer", 1, None, item_id, buffer_len=len(self.buffer))
            self.buffer_trace.append({"t": self.now, "buffer_len": len(self.buffer)})
            self._release_blocked()
            self._begin_item(1, item_id)

    # -- run ---------------------------------------------------------------
    def run(self) -> Dict[str, Any]:
        self._try_start(0)
        self._try_start(1)
        while self.event_q:
            ev = heapq.heappop(self.event_q)
            if ev.time > self.sim_time_s:
                break
            self.now = ev.time
            if ev.kind == "step_done":
                si, bi, item_id = ev.payload
                self._on_step_done(si, bi, item_id)
        for rt in self.stations:
            self._accrue(rt, self.sim_time_s)
        return self.summarize()

    # -- KPIs --------------------------------------------------------------
    def summarize(self) -> Dict[str, Any]:
        events = pd.DataFrame(self.events)
        readings = pd.DataFrame(self.readings)
        cycles = pd.DataFrame(self.station_cycles)
        completed = pd.DataFrame(self.completed)
        buffer_trace = pd.DataFrame(self.buffer_trace)

        n = len(completed)
        horizon_h = self.sim_time_s / 3600.0
        tput = n / horizon_h if horizon_h > 0 else 0.0
        takt = (self.sim_time_s / n) if n else float("nan")

        rows = []
        for si, rt in enumerate(self.stations):
            c = cycles[cycles["station"] == si]["cycle_s"] if len(cycles) else pd.Series(dtype=float)
            # Inter-pick interval at bin 1 — the pace the station is actually run at.
            if len(events):
                first_picks = events[
                    (events["event"] == "pick") & (events["station"] == si) & (events["bin"] == 0)
                ]["t"].sort_values()
                ip = first_picks.diff().dropna()
            else:
                ip = pd.Series(dtype=float)
            rows.append({
                "station": rt.spec.name,
                "units": int(len(c)),
                "nominal_cycle_s": rt.spec.nominal_cycle_s,
                "cycle_mean_s": float(c.mean()) if len(c) else float("nan"),
                "cycle_cv": float(c.std() / c.mean()) if len(c) > 1 and c.mean() else float("nan"),
                "inter_pick_mean_s": float(ip.mean()) if len(ip) else float("nan"),
                "busy_pct": 100.0 * rt.busy_s / self.sim_time_s,
                "blocked_pct": 100.0 * rt.blocked_s / self.sim_time_s,
                "starved_pct": 100.0 * rt.starved_s / self.sim_time_s,
            })
        station_stats = pd.DataFrame(rows)

        bottleneck = (
            station_stats.loc[station_stats["busy_pct"].idxmax(), "station"]
            if len(station_stats) else "-"
        )

        bin_rows = []
        for si, rt in enumerate(self.stations):
            for bi, b in enumerate(rt.spec.bins):
                bin_rows.append({
                    "station": rt.spec.name,
                    "bin": b.name,
                    "start_weight_g": b.start_weight_g,
                    "part_weight_g": b.part_weight_g,
                    "parts_per_pick": b.parts_per_pick,
                    "mean_step_s": b.mean_step_s,
                    "cv": b.cv,
                    "picks": rt.picks[bi],
                    "refills": rt.refills[bi],
                    "weight_now_g": rt.weights[bi],
                    "parts_left": rt.weights[bi] / b.part_weight_g if b.part_weight_g > 0 else float("nan"),
                })
        bin_stats = pd.DataFrame(bin_rows)

        return {
            "throughput_per_hour": float(tput),
            "completed": int(n),
            "takt_s": float(takt),
            "bottleneck_station": bottleneck,
            "lead_time_mean_s": float(completed["lead_s"].mean()) if n else float("nan"),
            "wip_end": int(self.item_counter - n),
            "stockouts": int((events["event"] == "stockout").sum()) if len(events) else 0,
            "station_stats": station_stats,
            "bin_stats": bin_stats,
            "events": events,
            "readings": readings,
            "cycles": cycles,
            "completed_df": completed,
            "buffer_trace": buffer_trace,
        }


def default_line() -> List[StationSpec]:
    """Two stations, five bins each. S2 is deliberately the slower station."""
    s1 = [
        BinSpec("S1_bin1", 500.0, 10.0, 1, 6.0, 0.25),
        BinSpec("S1_bin2", 400.0, 8.0, 2, 5.0, 0.20),
        BinSpec("S1_bin3", 700.0, 2.0, 8, 4.0, 0.30),
        BinSpec("S1_bin4", 600.0, 12.0, 1, 5.5, 0.18),
        BinSpec("S1_bin5", 300.0, 5.0, 4, 4.5, 0.22),
    ]
    s2 = [
        BinSpec("S2_bin1", 500.0, 10.0, 2, 7.0, 0.25),
        BinSpec("S2_bin2", 450.0, 15.0, 1, 6.5, 0.20),
        BinSpec("S2_bin3", 800.0, 4.0, 5, 5.5, 0.28),
        BinSpec("S2_bin4", 350.0, 7.0, 2, 6.0, 0.18),
        BinSpec("S2_bin5", 250.0, 5.0, 3, 5.0, 0.22),
    ]
    return [StationSpec("S1_assembly", s1), StationSpec("S2_finishing", s2)]
