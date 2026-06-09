import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

MODEL_PATH = "./model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

ID2TAG = {
    0: "O-Material_composite", 1: "B-Material_composite", 2: "I-Material_composite",
    3: "O-Process", 4: "B-Process", 5: "I-Process",
    6: "O-Condition", 7: "B-Condition", 8: "I-Condition",
    9: "O-Property_value", 10: "B-Property_value", 11: "I-Property_value"
}
ALLOWED_TYPES = ["Material_composite", "Process", "Property_value", "Condition"]

def extract_entities(text):
    tokens = text.strip().split()
    encoded = tokenizer(tokens, is_split_into_words=True, return_tensors="pt",
                        truncation=True, max_length=512)
    word_ids = encoded.word_ids()
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    predictions = torch.argmax(outputs.logits, dim=-1).squeeze().cpu().numpy()

    entities = []
    current_entity = ""
    current_type = ""
    previous_word_idx = None

    for i, word_idx in enumerate(word_ids):
        if word_idx is None or word_idx == previous_word_idx:
            continue
        label = ID2TAG[predictions[i]]
        if label.startswith("B-"):
            if current_entity and current_type in ALLOWED_TYPES:
                entities.append((current_entity.strip(), current_type))
            current_entity = tokens[word_idx]
            current_type = label.split("-")[1]
        elif label.startswith("I-") and current_type == label.split("-")[1]:
            current_entity += " " + tokens[word_idx]
        else:
            if current_entity and current_type in ALLOWED_TYPES:
                entities.append((current_entity.strip(), current_type))
            current_entity = ""
            current_type = ""
        previous_word_idx = word_idx

    if current_entity and current_type in ALLOWED_TYPES:
        entities.append((current_entity.strip(), current_type))

    if not entities:
        return "No entities found."

    result = ""
    grouped = {}
    for ent, typ in entities:
        grouped.setdefault(typ, []).append(ent)

    for typ, ents in grouped.items():
        result += f"**{typ}**\n"
        for e in ents:
            result += f"  • {e}\n"
        result += "\n"

    return result.strip()

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

demo.launch(allow_origins=["*"])