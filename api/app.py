from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib

app = FastAPI()

# permitir o site acessar a API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    text: str

model = joblib.load("models/spam_detection_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

@app.post("/predict/")
async def predict(message: Message):
    vectorized_message = vectorizer.transform([message.text])
    prediction = model.predict(vectorized_message)
    return {"spam": int(prediction[0])}

@app.get("/")
async def root():
    return {"status": "API online"}
