# MatSciBERT NER

A Named Entity Recognition pipeline for extracting structured information from materials science literature.

## What it does

Given a sentence or abstract from a materials science paper, the model identifies and labels:

- **Material / Composite** — named materials, compounds, composites
- **Process** — synthesis and fabrication methods
- **Condition** — experimental parameters (temperature, pressure, speed)
- **Property / Value** — measured outcomes and performance metrics

## Demo

Live app: https://huggingface.co/spaces/Shabuuuuuuuuuuu/matscibert-ner

## Model

Fine-tuned from [MatSciBERT](https://huggingface.co/m3rg-iitd/matscibert), a BERT model pre-trained on materials science literature. Adapted for token classification using a BIO tagging scheme across 12 label IDs.

Training data: 40 annotated materials science papers using a few-shot prompting strategy.

## Stack

- Model: HuggingFace Transformers + PyTorch
- Backend: Gradio on HuggingFace Spaces (free tier, CPU)
- Frontend: Vanilla HTML/CSS/JS hosted on GitHub Pages

## Usage

Paste any materials science sentence into the input box and click Extract Entities.

Example input:
Solvothermal synthesis of ZIF-8 MOF at 298 K shows high CO2 adsorption capacity of 4.5 mmol/g.
