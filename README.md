# MatSciBERT NER UI
This repository contains the beautiful frontend UI and the Gradio backend code for the **MatSciBERT Named Entity Recognition** model.

## What it does
Given a sentence or abstract from a materials science paper, the model identifies and labels:
- **Material / Composite** — named materials, compounds, composites
- **Process** — synthesis and fabrication methods
- **Condition** — experimental parameters (temperature, pressure, speed)
- **Property / Value** — measured outcomes and performance metrics

## Setup
- `index.html`: Open directly in browser or host on GitHub Pages.
- `app.py` & `requirements.txt`: Push to Hugging Face Spaces.
- `model/`: Make sure to upload your fine-tuned MatSciBERT safetensors and tokenizer files to the Hugging Face Space.
