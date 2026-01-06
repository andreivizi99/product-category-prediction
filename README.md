# Product Category Prediction

## Overview
This project implements a Machine Learning model that automatically predicts
the category of a product based on its title.

The goal is to reduce manual product classification effort and improve
efficiency in an online retail environment.

## Business Context
Online stores add thousands of new products every day.
Manual category assignment is time-consuming and error-prone.

This project demonstrates how Machine Learning can automate this process
and improve operational efficiency.

## Dataset
The dataset contains over 30,000 products with the following relevant fields:
- Product Title
- Category Label
- Merchant information and metadata

The dataset is stored in the `data/` folder.

## Project Structure
```
product-category-prediction/
├── data/
│   └── IMLP4_TASK_03-products (3).csv
├── models/
│   ├── product_category_model.pkl
│   └── tfidf_vectorizer.pkl
├── notebooks/
│   └── 01_exploration_and_model.ipynb
├── train_model.py
├── predict_category.py
└── README.md
```

## How to Run the Project

### 1. Train the model
Run the following command to train the model:
```bash
python train_model.py
```

This script:
- loads the dataset from the `data/` folder
- preprocesses the data
- trains the machine learning model
- saves the trained model and vectorizer in the `models/` folder

### 2. Predict product categories interactively
Run the interactive prediction script:
```bash
python predict_category.py
```

The user can enter a product title, and the model will predict
the corresponding product category.
Type `exit` to stop the program.

## Exploratory Analysis
The complete exploratory data analysis, feature engineering,
model experimentation, and evaluation are documented in the notebook:

`notebooks/01_exploration_and_model.ipynb`
