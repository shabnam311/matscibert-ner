# MatSciBERT NER UI
This repository contains the beautiful frontend UI and the Gradio backend code for the **MatSciBERT Named Entity Recognition** model.

## What it does
Given a sentence or abstract from a materials science paper, the model identifies and labels:
- **Material / Composite** — named materials, compounds, composites
- **Process** — synthesis and fabrication methods
- **Condition** — experimental parameters (temperature, pressure, speed)
- **Property / Value** — measured outcomes and performance metrics

## Architecture
The application is split into two parts:
1. **Frontend**: A highly optimized vanilla HTML/CSS/JS frontend hosted on GitHub Pages. It offers a stunning user experience with no build steps or heavy dependencies.
2. **Backend API**: The Gradio application (`app.py`) is deployed on Hugging Face Spaces. It loads the fine-tuned MatSciBERT model and exposes a REST API that the frontend communicates with.

### Flow Diagram
```mermaid
graph LR
    A[index.html (GitHub Pages)] -->|fetch() API call| B(HF Spaces API /run/predict)
    B -->|Returns structured JSON| A
    B --> C[(MatSciBERT Model)]
```

## API Usage
You can query the model programmatically using `fetch`:

```javascript
const response = await fetch(
  "https://shabuuuuuuuuuuu-matscibert-ner.hf.space/run/predict",
  {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ data: ["Graphene oxide nanosheets were reduced via chemical reduction."] })
  }
);
const result = await response.json();
console.log(result.data[0]);
```

## Setup
- `index.html`: Open directly in browser or host on GitHub Pages.
- `app.py` & `requirements.txt`: Push to Hugging Face Spaces.
- `model/`: Make sure to upload your fine-tuned MatSciBERT safetensors and tokenizer files to the Hugging Face Space.
