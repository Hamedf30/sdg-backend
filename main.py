from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import UploadFile, File
import pdfplumber
import io
from pydantic import BaseModel
from model_utils import predict, train_and_save, load_model
import pandas as pd

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://sdg-label-detector.vercel.app/"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextIn(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI is running!"}

@app.get("/ping")
def ping():
    return {"message": "pong"}

@app.post("/train")
def train():
    df = pd.read_csv('..\data\dataset.csv', sep='\t')
    texts = df['text'].fillna('').tolist()
    labels = df.get('labels_positive', df.get('labels_negative', pd.Series([0]*len(df)))).fillna(0).astype(int).tolist()
    clf, vec = train_and_save(texts, labels)
    return {"status": "trained", "n_samples": len(texts)}

@app.post("/predict")
def predict_text(payload: TextIn):
    res = predict(payload.text)
    return res

@app.post("/train")
def train():
    df = pd.read_csv('..\data\dataset.csv', sep='\t')
    texts = df['text'].fillna('').tolist()
    labels = df.get('labels_positive', df.get('labels_negative', pd.Series([0]*len(df)))).fillna(0).astype(int).tolist()
    clf, vec = train_and_save(texts, labels)
    return {"status": "trained", "n_samples": len(texts)}

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    # extract text from pdf
    content = await file.read()
    text = ""
    try:
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            for p in pdf.pages:
                text += p.extract_text() or ''
    except Exception:
        text = ''
    res = predict(text)
    return {"filename": file.filename, "result": res}