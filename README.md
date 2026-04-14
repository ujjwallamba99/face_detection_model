👤 Face Detection using HOG + SVM

A classical computer vision-based face detection system built using Histogram of Oriented Gradients (HOG) and Support Vector Machine (SVM). This project demonstrates how to detect faces in images using machine learning without deep learning.

📌 Project Overview

This project implements a face detection pipeline that:

Extracts features using HOG (Histogram of Oriented Gradients)
Trains a classifier using Linear SVM
Uses a sliding window approach to detect faces in images
Draws bounding boxes around detected faces

🧠 Key Concepts Used
HOG Feature Extraction
Supervised Learning (Binary Classification)
Sliding Window Technique
Image Preprocessing
Hyperparameter Tuning (GridSearchCV)

⚙️ Technologies & Libraries
Python 🐍
NumPy
Matplotlib
Scikit-learn
scikit-image
Joblib

📂 Dataset

✅ Positive Samples (Faces)
Dataset: Labeled Faces in the Wild (LFW)
Loaded using:
from sklearn.datasets import fetch_lfw_people
Shape: (13233, 62, 47)

❌ Negative Samples (Non-Faces)
Extracted from images like:
camera
text
coins
moon
page
coffee
chelsea
hubble deep field
Random patches generated using:
PatchExtractor
Shape: (30000, 62, 47)

🔄 Project Workflow

1️⃣ Feature Extraction (HOG)
Convert images to grayscale
Extract HOG features:
hog_vec, hog_vis = feature.hog(image, visualize=True)
Each image converted into a feature vector of size:
1215 features

2️⃣ Dataset Preparation
Combine:
Positive patches (faces)
Negative patches (non-faces)
X_train.shape = (43233, 1215)
Labels:
1 → Face
0 → Non-Face

3️⃣ Model Training
🔹 Baseline Model
Gaussian Naive Bayes
Accuracy ≈ 95–97%
🔹 Final Model
Linear Support Vector Classifier (LinearSVC)

Hyperparameter tuning:

GridSearchCV(LinearSVC(), {'C':[1.0,2.0,4.0,8.0]})

✅ Best Parameter:

C = 2.0

✅ Best Accuracy:

~98.85%

4️⃣ Face Detection (Sliding Window)
Image is scanned using small windows
Each patch is converted into HOG features
Model predicts whether patch contains a face
def sliding_window(...)
Bounding boxes drawn on detected faces
📸 Output
Input image is processed
Faces are highlighted using red bounding boxes
💾 Model Saving

The trained model is saved using Joblib:

import joblib
joblib.dump(model, "face_model.pkl")

▶️ How to Run the Project

1️⃣ Install Dependencies
pip install numpy matplotlib scikit-learn scikit-image joblib

2️⃣ Run the Code
Open Jupyter Notebook OR run script:
python main.py

3️⃣ Test on Your Image

Replace:

my_image = skimage.io.imread('my_photo.jpeg')

with your own image path.

⚠️ Common Errors (Important)
❌ Error:
UsageError: Line magic function `%matplotlin` not found

✅ Fix:

%matplotlib inline
❌ Feature Mismatch Error:
X has 2916 features, but model expects 1215

✅ Fix:

Ensure all images are resized to:
(62, 47)
Always apply HOG before prediction

🚀 Key Features
Lightweight (no deep learning required)
Works on CPU
End-to-end pipeline (training → detection → deployment-ready)
Custom image testing support

🚧 Future Improvements
Use CNN / Deep Learning (e.g., CNN, YOLO, MTCNN)
Real-time webcam detection
Improve accuracy with larger datasets
Deploy using Streamlit(but i have not deployed as i didn,t want to .)

📬 Author

Ujjwal Lamba
Aspiring Data Scientist & ML Engineer

⭐ If you like this project

Give it a ⭐ on GitHub and share it!
