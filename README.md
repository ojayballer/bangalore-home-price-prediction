#Bangalore Home Price Prediction

![Built with Python](https://img.shields.io/badge/Built%20with-Python-blue?style=flat&logo=python)
![Project Type](https://img.shields.io/badge/Project-Machine%20Learning-brightgreen)
![Deployed with Flask](https://img.shields.io/badge/Deployed%20With-Flask-black?logo=flask)
![Status](https://img.shields.io/badge/Status-Completed-blue)


This is a machine learning web application that predicts house prices in Bangalore based on key features such as location, area, number of bedrooms, bathrooms, and area type. It’s built using **Python**, **scikit-learn**, and **Flask** for deployment.

---

## 📊 Dataset

- **Source**: [Kaggle - Bengaluru House Price Data](https://www.kaggle.com/datasets/amitabhajoy/bengaluru-house-price-data)
- **Features Used**:
  - Location
  - Total square feet
  - BHK (bedrooms)
  - Bathrooms
  - Area type (Carpet, Super Built-up, etc.)

---

## ⚙️ What the Model Does

The backend uses a **Linear Regression model** trained on cleaned and preprocessed data to estimate the price of a house in lakhs. The pipeline includes:

- Handling missing values  
- Feature encoding (e.g., area type, location)  
- Outlier removal (e.g., unrealistic BHK per sqft ratios)  
- Feature engineering (price per sqft)  
- Hyperparameter tuning  
- Saving and reusing the model using `.pkl`

---

## 🖥 How It Works

- The **frontend** collects user inputs:
  - Area (in sqft)
  - Number of bedrooms (BHK)
  - Number of bathrooms
  - Area type
  - Location

- The **backend (Flask)**:
  - Loads the trained `.pkl` model
  - Processes the user inputs into the model format
  - Returns the predicted price in lakhs

---

##  What I Learned from This Project

- How to build and structure a complete machine learning pipeline  
- Dealing with real-world data: handling outliers and inconsistent entries  
- Deploying a model using **Flask** and connecting it to a live frontend  
- Organizing machine learning projects for deployment and portfolio use

---

> 💡This project showcases my ability to move beyond model training — to full end-to-end deployment and practical use.
