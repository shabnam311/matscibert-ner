import gradio as gr
from transformers import pipeline

MODEL_PATH = "./model"

try:
    ner_pipe = pipeline("token-classification", model=MODEL_PATH, aggregation_strategy="simple")
except Exception as e:
    ner_pipe = None
    startup_error = str(e)

def extract_entities(text):
    if ner_pipe is None:
        return f"**RUNTIME ERROR LOADING MODEL:**\n{startup_error}"
        
    try:
        entities = ner_pipe(text)
        if not entities:
            return "No entities found."

        result = ""
        grouped = {}
        for ent in entities:
            # entity group is like "Material_composite"
            typ = ent.get('entity_group', '')
            word = ent.get('word', '').strip()
            if word and typ:
                grouped.setdefault(typ, []).append(word)

        if not grouped:
            return "No entities found."

        for typ, ents in grouped.items():
            result += f"**{typ}**\n"
            for e in ents:
                result += f"  • {e}\n"
            result += "\n"

        return result.strip()
    except Exception as e:
        return f"**PREDICTION ERROR:**\n{str(e)}"

demo = gr.Interface(
    fn=extract_entities,
    inputs=gr.Textbox(
        lines=5,
        placeholder="Paste a materials science sentence or abstract...",
        label="Input Text"
    ),
    outputs=gr.Markdown(label="Extracted Entities"),
    title="MatSciBERT NER",
    description="Extracts Materials, Processes, Conditions, and Properties from scientific text.",
)

demo.launch()