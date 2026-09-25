"""app.py — Streamlit dashboard for the two-station line with per-bin load cells."""
import time
from typing import List

import numpy as np
import pandas as pd
import streamlit as st

from line_sim import BinSpec, StationSpec, BufferSpec, TwoStationLine, default_line

st.set_page_config(page_title="Two-Station Line — Load Cell Simulator", layout="wide")
st.title("Two-Station Assembly Line — Load Cell Simulator")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.header("Run settings")
sim_time_s = st.sidebar.slider("Simulation horizon (s)", 60, 43_200, 3600, 60)
seed = st.sidebar.number_input("Random seed", 0, 10_000, 42, 1)
buffer_cap = st.sidebar.number_input("Buffer capacity (S1 to S2)", 0, 50, 2, 1)
auto_refill = st.sidebar.checkbox(
    "Auto-refill empty bins", value=True,
    help="Off: a station stalls permanently when one of its bins runs dry.",
)

st.sidebar.divider()
st.sidebar.header("Bin configuration")

base = default_line()
station_specs: List[StationSpec] = []
for si, s in enumerate(base):
    with st.sidebar.expander(f"{s.name} — 5 bins", expanded=(si == 0)):
        bins: List[BinSpec] = []
        for bi, b in enumerate(s.bins):
            st.markdown(f"**Bin {bi + 1}**")
            c1, c2 = st.columns(2)
            sw = c1.number_input(
                "start weight (g)", 0.0, 1_000_000.0, float(b.start_weight_g), 10.0,
                key=f"sw{si}{bi}",
            )
            pw = c2.number_input(
                "part weight (g)", 0.1, 100_000.0, float(b.part_weight_g), 0.5,
                key=f"pw{si}{bi}",
            )
            c3, c4 = st.columns(2)
            pp = c3.number_input(
                "parts / pick", 1, 500, int(b.parts_per_pick), 1, key=f"pp{si}{bi}",
            )
            ms = c4.number_input(
                "mean step (s)", 0.5, 600.0, float(b.mean_step_s), 0.5, key=f"ms{si}{bi}",
            )
            cv = st.slider("CV", 0.0, 1.0, float(b.cv), 0.05, key=f"cv{si}{bi}")
            st.caption(
                f"holds {sw / pw:.0f} parts, {sw / (pw * pp):.0f} picks before empty"
                if pw > 0 else ""
            )
            bins.append(BinSpec(b.name, sw, pw, pp, ms, cv))
    station_specs.append(StationSpec(s.name, bins))

st.sidebar.divider()
run_btn = st.sidebar.button("Run simulation", type="primary", width="stretch")
reset_btn = st.sidebar.button("Reset", width="stretch")

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "res" not in st.session_state or reset_btn:
    st.session_state.res = None
    st.session_state.wall_s = None

if run_btn:
    sim = TwoStationLine(
        station_specs, BufferSpec(int(buffer_cap)),
        sim_time_s=float(sim_time_s), seed=int(seed), auto_refill=auto_refill,
    )
    t0 = time.time()
    with st.spinner("Running discrete-event engine..."):
        st.session_state.res = sim.run()
    st.session_state.wall_s = time.time() - t0

res = st.session_state.res
wall = st.session_state.wall_s

# ---------------------------------------------------------------------------
# KPI strip
# ---------------------------------------------------------------------------
k1, k2, k3, k4, k5, k6 = st.columns(6)
if res is None:
    for col, label in zip(
        [k1, k2, k3, k4, k5, k6],
        ["Throughput / hr", "Completed", "Takt (s)", "Bottleneck", "Lead time (s)", "Stockouts"],
    ):
        col.metric(label, "-")
    st.info("Set the bins in the sidebar, then run a simulation.")
else:
    k1.metric("Throughput / hr", f"{res['throughput_per_hour']:.1f}")
    k2.metric("Completed", f"{res['completed']}")
    k3.metric("Takt (s)", f"{res['takt_s']:.2f}" if np.isfinite(res["takt_s"]) else "-")
    k4.metric("Bottleneck", res["bottleneck_station"])
    k5.metric(
        "Lead time (s)",
        f"{res['lead_time_mean_s']:.1f}" if np.isfinite(res["lead_time_mean_s"]) else "-",
    )
    k6.metric("Stockouts", f"{res['stockouts']}", help=f"WIP at end: {res['wip_end']}")

st.divider()

tabs = st.tabs(["Production", "Load cells", "Buffer & WIP", "Timeline", "Config & export"])


def _trace(readings: pd.DataFrame, si: int, col: str) -> pd.DataFrame:
    """Step trace of one value per bin, forward-filled onto a common time axis."""
    sub = readings[readings["station"] == si]
    if sub.empty:
        return pd.DataFrame()
    wide = sub.pivot_table(index="t", columns="bin_name", values=col, aggfunc="last")
    return wide.sort_index().ffill()


# ---- Production ----------------------------------------------------------
with tabs[0]:
    if res is None:
        st.info("Run a simulation to see production results.")
    else:
        ss = res["station_stats"]
        st.subheader("Per-station results")
        st.dataframe(
            ss.style.format({
                "nominal_cycle_s": "{:.2f}", "cycle_mean_s": "{:.2f}", "cycle_cv": "{:.3f}",
                "inter_pick_mean_s": "{:.2f}", "busy_pct": "{:.1f}", "blocked_pct": "{:.1f}",
                "starved_pct": "{:.1f}",
            }),
            width="stretch", hide_index=True,
        )

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Cycle time: configured vs measured**")
            st.bar_chart(ss.set_index("station")[["nominal_cycle_s", "cycle_mean_s"]])
        with c2:
            st.markdown("**Where each station's time goes (%)**")
            st.bar_chart(ss.set_index("station")[["busy_pct", "blocked_pct", "starved_pct"]])

        st.markdown("**Cycle time vs inter-pick interval**")
        cmp_df = ss[["station", "cycle_mean_s", "inter_pick_mean_s"]].copy()
        cmp_df["takt_s"] = res["takt_s"]
        st.dataframe(
            cmp_df.style.format({
                "cycle_mean_s": "{:.2f}", "inter_pick_mean_s": "{:.2f}", "takt_s": "{:.2f}",
            }),
            width="stretch", hide_index=True,
        )
        st.caption(
            "Cycle time is finish minus start for one unit. Inter-pick is the gap between "
            "consecutive first-bin picks, which is what a parts-bin load cell alone can see. "
            "Inter-pick tracks the takt, not the station's own speed — the non-bottleneck "
            "station reads high because its waiting time is folded in."
        )

# ---- Load cells ----------------------------------------------------------
with tabs[1]:
    if res is None:
        st.info("Run a simulation to see load-cell readings.")
    else:
        view = st.radio(
            "Show", ["Weight (g)", "Parts remaining"], horizontal=True, label_visibility="collapsed",
        )
        col = "weight_g" if view == "Weight (g)" else "parts_est"

        for si, s in enumerate(station_specs):
            st.markdown(f"**{s.name} — load cells**")
            tr = _trace(res["readings"], si, col)
            if len(tr):
                st.line_chart(tr, height=260)
            else:
                st.caption("No readings.")

        st.markdown("**Bin summary**")
        bs = res["bin_stats"]
        st.dataframe(
            bs.style.format({
                "start_weight_g": "{:.1f}", "part_weight_g": "{:.2f}", "mean_step_s": "{:.2f}",
                "cv": "{:.2f}", "weight_now_g": "{:.1f}", "parts_left": "{:.1f}",
            }),
            width="stretch", hide_index=True,
        )
        st.caption("parts_left is weight_now_g divided by part_weight_g, the same way a load cell infers count.")

        ev = res["events"]
        sr = ev[ev["event"].isin(["stockout", "refill"])]
        st.markdown(f"**Stockouts and refills** ({len(sr)} events)")
        if len(sr):
            st.dataframe(
                sr[["t", "event", "station_name", "bin_name", "weight_g"]]
                .sort_values("t", ascending=False).head(200),
                width="stretch", hide_index=True, height=260,
            )
        else:
            st.caption("No bin ran dry during this run.")

# ---- Buffer & WIP --------------------------------------------------------
with tabs[2]:
    if res is None:
        st.info("Run a simulation to see buffer dynamics.")
    else:
        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown(f"**Buffer occupancy over time** (capacity {int(buffer_cap)})")
            bt = res["buffer_trace"].set_index("t")
            st.line_chart(bt, height=280)
            st.caption("Logged on every push and every pull, so this is the true occupancy.")
        with c2:
            st.markdown("**Lead time (s)**")
            cd = res["completed_df"]
            if len(cd):
                st.dataframe(
                    cd["lead_s"].describe().to_frame("lead_s").style.format("{:.2f}"),
                    width="stretch",
                )
            else:
                st.caption("No units completed.")

        st.markdown("**Cumulative completions**")
        if len(res["completed_df"]):
            cum = res["completed_df"][["t"]].copy()
            cum["completed"] = range(1, len(cum) + 1)
            st.line_chart(cum.set_index("t"), height=260)

# ---- Timeline ------------------------------------------------------------
with tabs[3]:
    if res is None:
        st.info("Run a simulation to view the event log.")
    else:
        ev = res["events"]
        kinds = st.multiselect(
            "Event types", sorted(ev["event"].unique()),
            default=["pick", "step_done", "station_start", "station_finish", "complete"],
        )
        n_show = st.slider("Rows", 50, 2000, 300, 50)
        sel = ev[ev["event"].isin(kinds)].sort_values("t").tail(n_show)
        cols = [c for c in ["t", "event", "station_name", "bin_name", "item_id",
                            "weight_g", "parts_taken", "buffer_len"] if c in sel.columns]
        st.dataframe(sel[cols], width="stretch", hide_index=True, height=520)

# ---- Config & export -----------------------------------------------------
with tabs[4]:
    st.subheader("Configuration")
    st.json({
        "sim_time_s": sim_time_s,
        "seed": int(seed),
        "buffer_capacity": int(buffer_cap),
        "auto_refill": auto_refill,
        "stations": [
            {"name": s.name, "nominal_cycle_s": s.nominal_cycle_s,
             "bins": [b.__dict__ for b in s.bins]}
            for s in station_specs
        ],
    })
    if res is not None:
        st.subheader("Export")
        st.caption(f"Run took {wall:.2f}s of wall clock." if wall else "")
        for label, key, fname in [
            ("Event log", "events", "events.csv"),
            ("Load-cell readings", "readings", "readings.csv"),
            ("Station cycles", "cycles", "cycles.csv"),
            ("Bin summary", "bin_stats", "bins.csv"),
        ]:
            df = res[key]
            st.download_button(
                f"Download {label} ({len(df)} rows)",
                df.to_csv(index=False).encode("utf-8"),
                file_name=fname, mime="text/csv", key=f"dl_{key}",
            )

st.divider()
st.caption("Two-station line with per-bin load cells | VIPR Project 5")
