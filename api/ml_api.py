# from fastapi import FastAPI
# from pydantic import BaseModel
# from api.emotion_model import EmotionModel
# from api.preprocessing import decode_and_preprocess
# from api.smoothing import Smoother

# app = FastAPI()

# model = EmotionModel()
# smoother = Smoother()

# class ImageData(BaseModel):
#     image: str

# print('ML API is running.')

# @app.post("/predict-cnn")
# def predict(data: ImageData):
#     x = decode_and_preprocess(data.image)
#     label, confidence = model.predict(x)
#     stable_label = smoother.update(label)

#     return {
#         "emotion": stable_label,
#         "confidence": confidence
#     }

from fastapi import FastAPI
from pydantic import BaseModel
from api.emotion_model import EmotionModel
from api.preprocessing import decode_and_preprocess
from api.smoothing import Smoother
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "https://mood-o-meter.onrender.com",
    "http://localhost:3000",  # if testing locally
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      
    allow_credentials=True,
    allow_methods=["*"],        # GET, POST, OPTIONS, etc.
    allow_headers=["*"],        # Content-Type, Authorization, etc.
)
model = EmotionModel()
smoother = Smoother()

class ImageData(BaseModel):
    image: str

@app.post("/predict-cnn")
def predict(data: ImageData):
    x = decode_and_preprocess(data.image)
    label, confidence = model.predict(x)
    stable_label = smoother.update(label)

    return {
        "emotion": stable_label,
        "confidence": confidence
    }
