# Scientific Search ABM

Agent-based model of collective conservatism in scientific topic choice.
MACSS 40550 — Agent-Based Modeling, Spring 2026.

## Files

- `model.py` — Model class: landscape construction, recognition, turnover, DVs
- `agents.py` — Scientist agent: decision rule (bounded search → threshold gate → weighted choice)
- `app.py` — Interactive Solara dashboard

## Running

```bash
pip install mesa[viz] networkx numpy
solara run app.py
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `recognition_bias` | 0.5 | How strongly the field favors legible work |
| `aspiration_bonus` | 0.4 | How much rbc lowers career threshold / career weight |
| `domestication_rate` | 0.03 | Speed at which explored topics gain recognizability |
| `turnover_rate` | 0.05 | Fraction of lowest-rep agents replaced each step |
