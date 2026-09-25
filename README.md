# Two-Station Assembly Line — Load Cell Simulator

A Streamlit dashboard for a discrete-event simulation of a two-station manual assembly line with per-bin load cells.

## Files

- `app.py` — Streamlit dashboard
- `line_sim.py` — discrete-event simulation engine
- `requirements.txt` — Python dependencies

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push these files to the root of a GitHub repository.
2. In Streamlit Community Cloud, create a new app.
3. Select the repository and branch.
4. Set the main file path to `app.py`.
5. Deploy.
