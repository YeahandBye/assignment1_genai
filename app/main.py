from fastapi import FastAPI
from pydantic import BaseModel
from app.embedding_model import calculate_embedding

app = FastAPI()


class EmbeddingRequest(BaseModel):
    text: str


@app.get("/")
def read_root():
    return {"message": "Embedding API is running"}


@app.post("/embedding")
def get_embedding(request: EmbeddingRequest):
    embedding = calculate_embedding(request.text)

    return {
        "text": request.text,
        "embedding": embedding.tolist()
    }
# CNN Image Classification API

import io
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from fastapi import UploadFile, File, HTTPException

from app.cnn_model import CNN


# CIFAR10 class names
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# Load trained CNN model
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "cnn_cifar10.pth"

device = torch.device("cpu")

cnn_model = CNN().to(device)
cnn_model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device, weights_only=True)
)
cnn_model.eval()

# Image preprocessing
image_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])


@app.post("/classify-image")
async def classify_image(file: UploadFile = File(...)):

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid image file"
        )

    image_tensor = image_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = cnn_model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_index = probabilities.argmax(dim=1).item()

    return {
        "filename": file.filename,
        "predicted_class": CLASS_NAMES[predicted_index],
        "confidence": round(
            probabilities[0, predicted_index].item(), 4
        )
    }

