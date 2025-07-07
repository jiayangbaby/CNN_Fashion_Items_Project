import os
import requests
import time

# API URL
url = "http://localhost:8501/predict"

# Path to the folder with test images
test_image_folder = r"C:\Users\jia.wang\Downloads\fashion-mnist-end-to-end-project-main\project\test_images"

def test_single_image(image_path):
    with open(image_path, "rb") as img_file:
        start = time.time()
        response = requests.post(url, files={"file": img_file})
        end = time.time()
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ {os.path.basename(image_path)} → {result['predicted_class']} ({result['confidence']*100:.2f}%) [{end-start:.2f}s]")
    else:
        print(f"❌ {os.path.basename(image_path)} → Failed with status {response.status_code}")

def batch_test():
    for filename in os.listdir(test_image_folder):
        if filename.endswith((".png", ".jpg", ".jpeg")):
            test_single_image(os.path.join(test_image_folder, filename))

if __name__ == "__main__":
    batch_test()

