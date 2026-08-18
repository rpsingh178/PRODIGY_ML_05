# PRODIGY_ML_05 — Food Recognition and Calorie Estimation

## 📌 Task 5

Develop a model that can accurately recognize food items from images and estimate their calorie content, enabling users to track their dietary intake and make informed food choices.

## 📖 Project Overview

This project uses a deep learning image classification model to recognize food items from images.

A pretrained **ResNet-18** model was fine-tuned to classify images into 11 different food categories. After identifying the food item, the system provides an approximate calorie estimate based on a standard 100-gram serving.

## 🍽️ Food Categories

The model recognizes the following 11 food categories:

- Apple Pie
- Cheesecake
- Chicken Curry
- French Fries
- Fried Rice
- Hamburger
- Hot Dog
- Ice Cream
- Omelette
- Pizza
- Sushi

## 📊 Dataset

The project uses the **Food-11 dataset**, containing:

- **9,900 training images**
- **1,100 testing images**
- **11 food categories**

