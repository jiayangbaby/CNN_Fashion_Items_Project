# test_api.py
import requests
import os

API_URL = "http://localhost:8501/predict"
#TEST_FOLDER  = r"C:\Users\jia.wang\Downloads\fashion-mnist-end-to-end-project-main\project\test_images\failure_test"
TEST_FOLDER  = r"C:\Users\jia.wang\Downloads\fashion-mnist-end-to-end-project-main\project\test_images"

def get_test_images():
    return [
        os.path.join(TEST_FOLDER, f)
        for f in os.listdir(TEST_FOLDER)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

def test_status_code_200():
    for path in get_test_images():
        with open(path, "rb") as f:
            response = requests.post(API_URL, files={"file": f})
        assert response.status_code == 200, f"{os.path.basename(path)} failed with {response.status_code}"

def test_contains_prediction_fields():
    for path in get_test_images():
        with open(path, "rb") as f:
            response = requests.post(API_URL, files={"file": f})
        data = response.json()
        assert "predicted_class" in data, f"{os.path.basename(path)} missing predicted_class"
        assert "confidence" in data, f"{os.path.basename(path)} missing confidence"

def test_confidence_range():
    for path in get_test_images():
        with open(path, "rb") as f:
            response = requests.post(API_URL, files={"file": f})
        confidence = response.json()["confidence"]
        assert 0.0 <= confidence <= 1.0, f"{os.path.basename(path)} has out-of-range confidence: {confidence}"

def test_invalid_file_fails():
    files = {"file": ("fake.txt", b"this is not an image")}
    response = requests.post(API_URL, files=files)
    assert response.status_code >= 400, "Expected failure for invalid file format"
