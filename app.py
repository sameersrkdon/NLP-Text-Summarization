from pathlib import Path
import logging
import re
import threading

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from transformers import T5ForConditionalGeneration, T5Tokenizer

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "saved_summary_model"
HOST = "127.0.0.1"
PORT = 5000
MAX_TEXT_CHARS = 15_000

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

if not MODEL_DIR.is_dir():
    raise FileNotFoundError(f"Trained model folder not found: {MODEL_DIR}")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOKENIZER = T5Tokenizer.from_pretrained(str(MODEL_DIR))
MODEL = T5ForConditionalGeneration.from_pretrained(str(MODEL_DIR)).to(DEVICE)
MODEL.eval()
MODEL_LOCK = threading.Lock()
DEVICE_NAME = torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "CPU"

app = FastAPI(
    title="Text Summarizer",
    description="Summarize conversations with the locally trained T5 model.",
    version="1.0.0",
)
templates = Jinja2Templates(directory=str(PROJECT_DIR))


class DialogueInput(BaseModel):
    dialogue: str = Field(min_length=1, max_length=MAX_TEXT_CHARS)


def clean_data(text: str) -> str:
    text = re.sub(r"\r\n", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    return text.strip().lower()


def summarize_dialogue(dialogue: str) -> str:
    dialogue = clean_data(dialogue)
    if not dialogue:
        raise HTTPException(status_code=400, detail="Paste a conversation first.")

    inputs = TOKENIZER(
        dialogue,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt",
    )
    inputs = {key: value.to(DEVICE) for key, value in inputs.items()}

    try:
        with MODEL_LOCK, torch.inference_mode():
            generated = MODEL.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=150,
                num_beams=4,
                early_stopping=True,
            )
    except Exception as exc:
        logging.exception("Summary generation failed")
        raise HTTPException(status_code=500, detail="The model could not generate a summary.") from exc

    return TOKENIZER.decode(generated[0], skip_special_tokens=True).strip()


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"device_name": DEVICE_NAME},
    )


@app.post("/summarize/")
def summarize(dialogue_input: DialogueInput):
    summary = summarize_dialogue(dialogue_input.dialogue)
    return {"summary": summary, "device": DEVICE_NAME}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT)
