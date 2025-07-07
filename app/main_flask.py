from flask import Flask, request, jsonify, send_from_directory
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import csv
from datetime import datetime, timezone
from io import BytesIO
from PIL import Image, UnidentifiedImageError, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # optional, allows loading damaged files
from flask_cors import CORS
import boto3
from functools import wraps
from dotenv import load_dotenv


#API protetion with token
load_dotenv()
API_TOKEN = os.getenv("UPLOAD_API_TOKEN")

def requires_token(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get("Authorization", "")
        token = auth_header.replace("Bearer ", "").strip()
        if token != API_TOKEN:
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated



app = Flask(__name__)
CORS(app)

working_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(working_dir, "logs"), exist_ok=True)
model_path = f"{working_dir}/trained_model/trained_fashion_mnist_model.h5"
frontend_dir = os.path.abspath(os.path.join(working_dir, "..", "frontend"))

today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
csv_file = os.path.join(working_dir, "logs", f"user_events_{today}.csv")


# Load the pre-trained model
#model = tf.keras.models.load_model(model_path)
model = tf.keras.models.load_model(model_path, compile=False)

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])



# Define class labels for Fashion MNIST dataset
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# Function to preprocess the uploaded image
def preprocess_image(image):
    img = Image.open(image)
    img = img.resize((28, 28))
    img = img.convert('L')  # Convert to grayscale for the frist version
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape((1, 28, 28, 1))
    return img_array

# Event logger helper function, called in the flask predict
def log_event_to_csv(data):
    data["server_timestamp"] = datetime.now(timezone.utc).isoformat()
    file_exists = os.path.isfile(csv_file)

    with open(csv_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)

    print(f"📘 Logged event: {data}")

# S3 upload function helper
def upload_to_s3(file_path, bucket_name, object_name):
    s3 = boto3.client("s3")
    s3.upload_file(file_path, bucket_name, object_name)

# Health check
@app.route("/ping")
def health_check():
    return "API is running!"

# Serve index.html
@app.route("/")
def serve_index():
    return send_from_directory(frontend_dir, "index.html")

# Serve static assets (JS, CSS, etc.)
@app.route("/<path:filename>")
def serve_static(filename):
    return send_from_directory(frontend_dir, filename)



@app.route("/predict", methods=["POST"])

def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]
        image_bytes = file.read()
        image_stream = BytesIO(image_bytes)
        #image_stream = Image.open(BytesIO(image_bytes)).convert("L")

        # Debug print: whether file is received
        print(f"Received file: {file.filename}, {len(image_bytes)} bytes")

        input_tensor = preprocess_image(image_stream)

        preds = model.predict(input_tensor)[0]
        probs = tf.nn.softmax(preds).numpy()
        pred_label = class_names[np.argmax(probs)]
        confidence = float(np.max(probs))

        log_event_to_csv({
            "filename": file.filename,
            "predicted_class": pred_label,
            "confidence": confidence
        })
        
        #auto-upload to s3
        try:
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            csv_file = os.path.join(working_dir, "logs", f"user_events_{today}.csv")

            upload_to_s3(
                file_path=csv_file,
                bucket_name="sagemaker-demo-bucket-jiayang",
                object_name=f"logs/user_events_{today}.csv"
            )
            print("Sucess: Auto-uploaded log to S3")
        except Exception as e:
            print("Failed: Auto-upload failed:", e)


        return jsonify({
            "predicted_class": pred_label,
            "confidence": confidence
        })
    
    except (UnidentifiedImageError, OSError) as e:
        return jsonify({"error": f"Invalid image file: {str(e)}"}), 422

    except Exception as e:
        print("ERROR:", e)
        return jsonify({"error": str(e)}), 500
    
#write the user activities if it is a new day
@app.route("/log", methods=["POST"])
def log_event():
    try:
        data = request.json
        data["server_timestamp"] = datetime.now(timezone.utc).isoformat()

        file_exists = os.path.isfile(csv_file)
        with open(csv_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=data.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(data)

        print(f"Sucess: Logged event: {data}")
        return jsonify({"status": "logged"}), 200

    except Exception as e:
        print("Logging error:", e)
        return jsonify({"error": str(e)}), 500

#  S3 trigger endpoint, mannual control
@app.route("/upload-log", methods=["POST"])
@requires_token
def trigger_s3_upload():
    try:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        upload_to_s3(
            file_path=csv_file,
            bucket_name="sagemaker-demo-bucket-jiayang",  
            object_name=f"logs/user_events_{today}.csv"
        )
        return jsonify({"status": "uploaded to S3 successfully"}), 200
    except Exception as e:
        print("Upload error:", e)
        return jsonify({"error": str(e)}), 500
   
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8501)
