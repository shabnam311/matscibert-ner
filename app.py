import gradio as gr
from transformers import pipeline
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import io
import docx
import PyPDF2
import os

app = FastAPI()

# Enable CORS for the GitHub Pages frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "./model"

try:
    if os.path.exists(MODEL_PATH):
        ner_pipe = pipeline("token-classification", model=MODEL_PATH, aggregation_strategy="simple")
    else:
        # Fallback to the hub if model isn't downloaded locally
        ner_pipe = pipeline("token-classification", model="m3rg-iitd/matscibert", aggregation_strategy="simple")
except Exception as e:
    ner_pipe = None
    startup_error = str(e)

class TextRequest(BaseModel):
    text: str

def parse_entities(entities_list):
    grouped = {}
    for ent in entities_list:
        typ = ent.get('entity_group', '').upper()
        word = ent.get('word', '').strip()
        if not word or not typ: continue
        
        if 'MAT' in typ or 'COMP' in typ:
            mapped_typ = 'Material_composite'
        elif 'PROC' in typ or 'METH' in typ:
            mapped_typ = 'Process'
        elif 'COND' in typ or 'TEMP' in typ:
            mapped_typ = 'Condition'
        elif 'PROP' in typ or 'VAL' in typ:
            mapped_typ = 'Property_value'
        elif 'LABEL_0' in typ or 'LABEL_1' in typ or 'LABEL_2' in typ or 'LABEL_3' in typ:
            # Fallback for base models
            mapping = {'LABEL_0': 'Material_composite', 'LABEL_1': 'Process', 'LABEL_2': 'Condition', 'LABEL_3': 'Property_value'}
            mapped_typ = mapping.get(typ, 'Material_composite')
        else:
            # If we don't know what it is, just guess material
            mapped_typ = 'Material_composite'
            
        grouped.setdefault(mapped_typ, []).append(word)
    
    result = ""
    if not grouped:
        return ""
        
    for typ, ents in grouped.items():
        result += f"**{typ}**\n"
        for e in ents:
            result += f"  • {e}\n"
        result += "\n"
    return result.strip()

@app.post("/api/extract_text")
async def extract_text_api(req: TextRequest):
    if ner_pipe is None:
        return {"error": f"RUNTIME ERROR LOADING MODEL: {startup_error}"}
    try:
        entities = ner_pipe(req.text)
        formatted = parse_entities(entities)
        return {"result": formatted}
    except Exception as e:
        return {"error": str(e)}

@app.post("/api/extract_file")
async def extract_file_api(file: UploadFile = File(...)):
    if ner_pipe is None:
        return {"error": f"RUNTIME ERROR LOADING MODEL: {startup_error}"}
    
    text = ""
    try:
        content = await file.read()
        
        # Parse based on extension
        if file.filename.endswith(".txt"):
            text = content.decode("utf-8", errors="ignore")
        elif file.filename.endswith(".pdf"):
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        elif file.filename.endswith(".docx"):
            doc = docx.Document(io.BytesIO(content))
            for para in doc.paragraphs:
                text += para.text + "\n"
        else:
            return {"error": "Unsupported file format. Please upload .txt, .pdf, or .docx"}
            
        if not text.strip():
            return {"error": "Could not extract text from file or file is empty."}
            
        # Truncate text if too long to prevent model OOM (MatSciBERT max seq len is 512, but pipeline handles chunks if configured, we'll just take first ~2000 chars for safety)
        if len(text) > 3000:
            text = text[:3000]
            
        entities = ner_pipe(text)
        formatted = parse_entities(entities)
        return {"text": text.strip(), "result": formatted}
        
    except Exception as e:
        return {"error": f"Failed to process file: {str(e)}"}

# Keep the original Gradio interface available at the root
def extract_entities_gradio(text):
    if ner_pipe is None:
        return f"**RUNTIME ERROR LOADING MODEL:**\n{startup_error}"
    try:
        entities = ner_pipe(text)
        formatted = parse_entities(entities)
        return formatted if formatted else "No entities found."
    except Exception as e:
        return f"**PREDICTION ERROR:**\n{str(e)}"

demo = gr.Interface(
    fn=extract_entities_gradio,
    inputs=gr.Textbox(lines=5, placeholder="Paste a materials science sentence or abstract...", label="Input Text"),
    outputs=gr.Markdown(label="Extracted Entities"),
    title="MatSciBERT NER",
    description="Extracts Materials, Processes, Conditions, and Properties from scientific text.",
)

app = gr.mount_gradio_app(app, demo, path="/")