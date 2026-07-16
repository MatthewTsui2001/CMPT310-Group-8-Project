Dog Breed & Age Classifier

A TensorFlow-based deep learning project that classifies a dog's breed and estimates its age from an image.

Overview

This project uses convolutional neural networks (CNNs) to perform two tasks from a single input image:


Breed Classification — Identifies the dog's breed from a set of known breed categories.
Age Estimation — Predicts the approximate age (or age group, e.g. puppy / adult / senior) of the dog.


The model can be trained as two separate networks or as a single multi-output network sharing a common feature extractor.

Features


Image preprocessing and augmentation pipeline (resizing, normalization, flips, rotations)
Transfer learning support using pretrained backbones (e.g. MobileNetV2, ResNet50, EfficientNet)
Multi-output model architecture: breed classification (softmax) + age prediction (regression or classification)
Training, validation, and evaluation scripts
Inference script for single-image predictions
Model checkpointing and TensorBoard logging


Project Structure

dog-breed-age-classifier/
├── data/
│   ├── raw/                # Original images
│   ├── processed/          # Preprocessed/augmented data
│   └── labels.csv          # Breed + age annotations
├── models/
│   └── saved_model/         # Trained model checkpoints
├── notebooks/
│   └── exploration.ipynb    # EDA and prototyping
├── src/
│   ├── data_loader.py        # Dataset loading and preprocessing
│   ├── model.py               # Model architecture definitions
│   ├── train.py               # Training script
│   ├── evaluate.py            # Evaluation script
│   └── predict.py             # Inference on new images
├── requirements.txt
└── README.md

Requirements


Python 3.9+
TensorFlow 2.x
NumPy, Pandas
Matplotlib / Seaborn (for visualization)
scikit-learn (for metrics and data splitting)


Install dependencies:

bashpip install -r requirements.txt

Dataset

This project expects a dataset of dog images labeled with:


Breed (categorical label)
Age (numeric value or age category, e.g. puppy, young, adult, senior)


Example datasets to consider:


Stanford Dogs Dataset (breed labels only — age labels would need to be added separately)
A custom-labeled dataset with both breed and age annotations


Place your dataset in data/raw/ and update data/labels.csv with corresponding breed and age labels.

Usage

1. Preprocess the data

bashpython src/data_loader.py --input_dir data/raw --output_dir data/processed

2. Train the model

bashpython src/train.py --data_dir data/processed --epochs 50 --batch_size 32

3. Evaluate the model

bashpython src/evaluate.py --model_path models/saved_model --data_dir data/processed

4. Run inference on a new image

bashpython src/predict.py --model_path models/saved_model --image_path path/to/dog.jpg

Example output:

Predicted Breed: Golden Retriever (94.2% confidence)
Predicted Age: Adult (approx. 3-5 years)

Model Architecture

The model uses a shared convolutional backbone (transfer learning from ImageNet-pretrained weights) with two output heads:


Breed head: Dense layer with softmax activation over N breed classes
Age head: Dense layer with either:

A single linear output (regression, predicting age in years), or
A softmax output over age categories (classification)





Evaluation Metrics


Breed classification: Accuracy, Precision, Recall, F1-score, Confusion Matrix
Age estimation: Mean Absolute Error (MAE) if regression, or Accuracy/F1 if classification


Future Improvements


Expand dataset with more breeds and age-labeled samples
Experiment with ensemble models
Add a web or mobile front-end for real-time predictions
Incorporate additional attributes (weight, size) for improved age estimation


License

MIT License — feel free to use and modify this project.

Acknowledgments


TensorFlow / Keras documentation and tutorials
Stanford Dogs Dataset contributors
