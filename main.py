from fastapi import FastAPI, File, UploadFile, Request
from typing import List
import shutil
import os
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from model import predict

app = FastAPI()

# Create a directory to store uploaded images
UPLOADS_DIRECTORY = "uploaded_images"
os.makedirs(UPLOADS_DIRECTORY, exist_ok=True)

# Mount the directory to serve static files
app.mount("/static", StaticFiles(directory=UPLOADS_DIRECTORY), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from any origin
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all HTTP headers
)

@app.get("/")
async def index():
    return {"Working": "Chal Raha hu bhai!!"}

@app.post("/upload-image/")
async def upload_video(files: List[UploadFile] = File(...)):
    saved_file_paths = []
    
    # Save uploaded files and print their paths
    for file in files:
        file_path = os.path.join(UPLOADS_DIRECTORY, "uploaded_image.jpg")
        saved_file_paths.append(file_path)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print("Uploaded image saved at:", file_path)
        response = predict(file_path)
    
    # Return the file paths of saved images
    return {"output": "/static/predicted_image.jpg"}