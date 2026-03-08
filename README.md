# 🌟 PhalFresh – AI Fruit Freshness Detection System
<p align="center"> <img src="https://dummyimage.com/1200x300/4CAF50/ffffff&text=PhalFresh:+AI+Fruit+Freshness+Detector" alt="PhalFresh Banner"/> </p> <p align="center"> <b>AI-powered fruit type recognition, freshness detection, and shelf-life prediction – all from a single image.</b> </p>

## 🏷️ Badges
<p align="center"> <img src="https://img.shields.io/badge/Build-Passing-brightgreen?style=flat-square"/> <img src="https://img.shields.io/badge/Accuracy-97%25--99%25-blue?style=flat-square"/> <img src="https://img.shields.io/badge/Framework-TensorFlow-orange?style=flat-square"/> <img src="https://img.shields.io/badge/API-HuggingFace-yellow?style=flat-square"/> <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square"/> <img src="https://img.shields.io/badge/Python-3.8+-blue?style=flat-square"/> </p>

## 🥭 What is PhalFresh?

PhalFresh is an AI-driven computer vision model built to:

✔ Detect fruit type

✔ Identify whether fruit is fresh or rotten

✔ Predict remaining shelf life

✔ Deliver real-time predictions (<1 sec)

✔ Perform reliably even under low-light and mobile camera conditions

Built with CNNs, TensorFlow, OpenCV, and HuggingFace, it is optimized for real-world performance and ease of use.

## 🚀 Features at a Glance

🎯 97–99% Accuracy for fruit type classification

🍏 95–98% Accuracy for fresh/rotten detection

⏱️ Real-time inference via HuggingFace (<1s)

📱 Robust on mobile images & low-light scenarios

🧮 Shelf-life prediction based on ripening trends

🌐 Simple web interface for image uploads


## 🧠 How the Model Works

🔹 1. Preprocessing

Resizing

Normalization

Lighting adjustments

Augmentations to improve robustness

🔹 2. CNN Model

A custom-trained CNN that learns:

Texture differences

Color decay patterns

Rot and mold structure

Ripening characteristics

🔹 3. Prediction Pipeline

Fruit Type → Fresh/Rotten → Shelf-Life Estimation

Returns results in <1 second

🔹 4. Web Interface

A clean UI for uploading/capturing images.

