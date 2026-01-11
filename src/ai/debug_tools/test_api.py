import pytest
from fastapi.testclient import TestClient
from icr_prototype import app
import numpy as np
import cv2
import io
from PIL import Image

client = TestClient(app)

def test_ping():
    response = client.get("/ping")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_recognize_empty():
    # Send an truly empty file
    files = {"file": ("empty.png", b"", "image/png")}
    response = client.post("/recognize", files=files)
    assert response.status_code == 200
    assert response.json()["status"] == "error"
    assert "empty" in response.json()["message"]

def test_recognize_invalid_bytes():
    # Send some random bytes that are not an image
    files = {"file": ("random.txt", b"this is not an image", "text/plain")}
    response = client.post("/recognize", files=files)
    assert response.status_code == 200
    assert response.json()["status"] == "error"
    # imdecode should return None for invalid data
    assert "Could not decode image" in response.json()["message"]

def create_test_image():
    # Create a small white image
    img = Image.new('RGB', (100, 30), color = (255, 255, 255))
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    return img_byte_arr.getvalue()

def test_recognize_valid_dummy():
    # This might take a moment as it loads the model
    img_bytes = create_test_image()
    files = {"file": ("test.png", img_bytes, "image/png")}
    response = client.post("/recognize", files=files)
    assert response.status_code == 200
    # Even if it returns empty text for a white image, it should not crash
    assert response.json()["status"] == "success"

def test_preprocess_empty():
    files = {"file": ("empty.png", b"", "image/png")}
    response = client.post("/preprocess", files=files)
    assert response.status_code == 200
    assert response.json()["status"] == "error"

def test_transcribe_empty():
    files = {"file": ("empty.wav", b"", "audio/wav")}
    response = client.post("/transcribe", files=files, data={"language": "de"})
    assert response.status_code == 200
    assert response.json()["status"] == "error"
