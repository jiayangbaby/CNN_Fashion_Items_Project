Fashion Image Classifier – End-to-End Deep Learning Pipeline
This repository presents an end-to-end image classification system for fashion items, originally based on the Fashion MNIST dataset. The project started from the tutorial here, but has since been significantly extended to improve both model performance and system design.

Project Summary:
1. A custom-trained CNN model that supports colored, high-resolution images.
2. A dual interface system: a lightweight Streamlit prototype and a production-ready Flask web app.
3. RESTful API services for real-time predictions and behavioral logging. Authenticated API access using token-based authentication.
4. A logging and analytics pipeline that automatically records user input and results to AWS S3 on a daily basis.

Key Enhancements:
1. Redesigned CNN architecture with additional convolutional and dense layers.
2. Support for higher image resolution and RGB input.
3. Trained on an expanded dataset beyond the original MNIST, improving generalization.
4. Extended training cycles (more epochs) and refined tensor preprocessing pipeline.

System Architecture:
1. Flask-based frontend with HTML + Vue for custom UI and smooth integration with the backend.
2. Streamlit version preserved for quick testing and demos.
3. Dockerized app for consistent deployment across environments.
4. Modular backend API for image classification and behavior logging.
5. Logging pipeline stores each user upload, prediction result, and timestamp in AWS S3, updated daily.

Deployment: 
1. The full system is containerized using Docker.
2. Flask app exposes the /predict endpoint for inference.
3. Behavior logs are stored in a structured format and uploaded to S3 via automated jobs.

Future Work:
Model retraining loop based on user feedback

Acknowledgment:
This project builds on the tutorial by CodeBasics (video), and expands it into a scalable, full-stack ML system with production-level capabilities.

