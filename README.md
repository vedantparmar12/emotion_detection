# Emotion Detection with Hugging Face Transformers
This repository implements an emotion detection model using Hugging Face Transformers and a variety of image processing techniques.

Table of Contents
Overview
Installation
Usage
Dataset
Results
Training Process
Contributing
Overview
This project leverages the power of Hugging Face Transformers, a state-of-the-art library for working with deep learning models in natural language processing (NLP) and computer vision. The model has been trained on a dataset of facial images to recognize seven distinct emotions:

Sad
Disgust
Angry
Neutral
Fear
Surprise
Happy
Installation
Clone this repository:

Bash
```
git clone https://github.com/vedantparmar12/emotion-detection.git
cd emotion-detection
```
Use code with caution.
content_copy
Install dependencies:

Bash
```
pip install -r requirements.txt
```
Use code with caution.
content_copy
(Ensure PyTorch is installed with CUDA support if you want to leverage GPU acceleration.)

Usage
Prepare your dataset: Organize your images into folders named after the corresponding emotions.
Edit the configuration file: Modify config.py to point to your dataset directory and adjust any training parameters.
Run the training script:
Bash
```
python train.py
```
Use code with caution.
content_copy
Inference: Use infer.py to load the trained model and predict emotions on new images.
Dataset
Dataset: Describe the dataset used for training and evaluation (e.g., name, size, source).
Results link:
```
https://docs.google.com/spreadsheets/d/1yiVR2MO70st1NFXqhWsk64VtHxPkdQrCSDXfXE8reOU/edit?gid=536768246#gid=536768246
```
Classification Report
Emotion	Precision	Recall	F1-Score	Support
Sad	0.8201	0.7893	0.8044	3407
Disgust	0.7005	0.5007	0.5840	1490
Angry	0.7855	0.7999	0.7926	2898
Neutral	0.9257	0.9192	0.9225	10657
Fear	0.8114	0.9054	0.8558	10557
Surprise	0.8256	0.7497	0.7858	5209
Happy	0.8411	0.8029	0.8216	3730

drive_spreadsheet
Export to Sheets
Accuracy
Overall accuracy: 0.8434

Macro/Weighted Averages
Macro avg: 0.8157 (precision), 0.7810 (recall), 0.7952 (f1-score)
Weighted avg: 0.8428 (precision), 0.8434 (recall), 0.8414 (f1-score)
Training Process
The model was trained using the following key libraries and techniques:

Hugging Face Transformers: The core library for model selection, fine-tuning, and inference.
PyTorch: Deep learning framework providing flexibility and efficiency.
Scikit-learn: For evaluation metrics and data splitting.
OpenCV (cv2): Image loading and basic processing.
Image Transformations: (CenterCrop, Normalize, RandomRotation, etc.) To augment the data and improve model robustness.
Transfer Learning: Leveraging pre-trained models to accelerate training and achieve better results.
Contributing
Contributions are welcome! Please feel free to open issues or submit pull requests.




tune

share


more_vert
