from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from api.emotion_model import EmotionModel
from api.preprocessing import decode_and_preprocess
from api.smoothing import Smoother

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://mood-o-meter.onrender.com",
        "http://localhost:3000"
    ],
    allow_credentials=False,   # IMPORTANT
    allow_methods=["*"],
    allow_headers=["*"],
)

model = EmotionModel()
smoother = Smoother()

class ImageData(BaseModel):
    image: str

@app.post("/predict-cnn")
async def predict(data: ImageData):
    x = decode_and_preprocess(data.image)
    label, confidence = model.predict(x)
    stable_label = smoother.update(label)

    return {
        "emotion": stable_label,
        "confidence": confidence
    }
    
