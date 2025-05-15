from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from ultralytics import YOLO
from PIL import Image
import io

app = FastAPI()

# Load your trained YOLO model
model = YOLO("best.pt")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read the uploaded image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # Run YOLO model on image
    results = model(image)

    # Draw boxes on the image
    image_with_boxes = results[0].plot()  # returns numpy array

    # Convert numpy array (OpenCV image) to PIL Image
    image_pil = Image.fromarray(image_with_boxes)

    # Save image to in-memory buffer
    buffer = io.BytesIO()
    image_pil.save(buffer, format="JPEG")
    buffer.seek(0)

    # Return image as HTTP response
    return StreamingResponse(buffer, media_type="image/jpeg")





@app.get("/fetch")
async def fetchMessage():
    return {"message": "Hello World"}
