from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io
import torch
from torchvision import models, transforms

app = FastAPI()

# Load the pre-trained PyTorch model
model = models.resnet50(pretrained=True)
model.eval()

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define the prediction route
@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    # Read the image file
    img_bytes = await image.read()
    img = Image.open(io.BytesIO(img_bytes))

    # Preprocess the image
    img = transform(img).unsqueeze(0)

    # Make the prediction
    with torch.no_grad():
        output = model(img)

    # Process the prediction
    _, predicted_idx = torch.max(output, 1)
    predicted_label = predicted_idx.item()

    return {"predicted_label": predicted_label}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=5000)