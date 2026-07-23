# Nanoparticle Tumor Targeting Simulator

Interactive 2D stochastic simulation of nanoparticle transport, receptor binding,
unbinding, internalization, and clearance.

## Visual outputs
- Animated particle map
- Free, bound, internalized, and cleared particle states
- Tumor outline
- Individual trajectories
- State kinetics
- Tumor accumulation curves
- Downloadable GIF and CSV data

## Run
    python -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt
    streamlit run app.py

On macOS/Linux use:

    source .venv/bin/activate

## Scope
Reduced-order portfolio model. Not a validated clinical or pharmacokinetic tool.
