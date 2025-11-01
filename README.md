# fruit-classifier-cnn
CNN-based fruit image classifier trained on 12k apple, grape, raspberry and peach images.

Fruit Classifier – ResNet/EfficientNet CNN

Overview

This project is a computer vision classifier trained on a self-collected dataset of roughly 12,000 images of apples, grapes, raspberries and peaches captured under varying visual conditions (ideal, in-context, and challenging).
The goal was to build and deploy a convolutional neural network (CNN) capable of identifying fruit type across lighting, angle, and background variations.

The model was trained in Python using ResNet-50 and EfficientNet-B0, exported to ONNX, and deployed through a static web demo powered by ONNX Runtime Web, enabling client-side inference directly in the browser (no server required).




Model Training
	•	Architecture: EfficientNet-B0
	•	Dataset: ~12,000 self-captured fruit images (3,000 per class)
	•	Training approach: single prototype run to verify deployment pipeline
	•	Accuracy (approx.): 80–90 % top-1 performance on in-sample testing
	•	Frameworks: PyTorch → ONNX export for inference

NOTE: The original training environment has since been reset.
The notebook (CV.ipynb) reflects the final working configuration and code used to produce the exported ONNX model, but may not re-execute as-is.

Live Demo

The interactive classifier demo runs entirely in the browser via ONNX Runtime Web.
Users can upload a sample image to see predicted fruit type and confidence.

View Demo on Github Pages: https://derekklue.github.io/fruit-classifier-cnn/

Results

Top 1% Accuracy: ~90%
Classes: Apple, Grape, Raspberry, Peach
Dataset Size: 12,000 Images

Future Improvements
	•	Recreate training environment for reproducible experiments
	•	Add 70/20/10 train/validation/test split
	•	Log metrics: accuracy, precision, recall, confusion matrix
	•	Optimize ONNX model for smaller browser footprint

Acknowledgment

Color palette and initial UI layout were co-designed as part of a group collaboration; this static version reflects my independent CNN development, training, and deployment work.


Summary

This project demonstrates an end-to-end ML workflow — from dataset creation and CNN training to browser-based deployment — while maintaining a lightweight, fully static implementation suitable for GitHub Pages hosting.

