# Customer Churn Prediction Web App

This project is a **Customer Churn Prediction** application that predicts whether a customer is likely to leave a bank based on their profile and account information. It uses an **Artificial Neural Network (ANN)** trained on the `Churn_Modelling.csv` dataset and is deployed as an interactive **Streamlit** web app.

---

## Live Demo

Try the interactive Streamlit web app here: [Customer Churn Prediction App](https://ann-classification-churn-pr524fbkdxg2hom9szegyx.streamlit.app/)

---

## Table of Contents

- [Project Overview](#project-overview)  
- [Dataset](#dataset)  
- [Features](#features)   
- [Installation](#installation)   
- [Project Structure](#project-structure)   

---

## Project Overview

Customer churn is a critical metric for businesses. Banks want to identify which customers are likely to leave so they can take proactive actions to retain them.  

This project includes:

- Data preprocessing (handling categorical variables and scaling features).  
- Training an ANN model to predict churn.  
- Saving the trained model and preprocessing objects using **Pickle**.  
- Deploying the model in a **Streamlit web application** for real-time predictions.  

---

## Dataset

The dataset used is `Churn_Modelling.csv` and contains **10,000 customer records** with the following features:

| Column | Description |
|--------|-------------|
| RowNumber | Unique row ID |
| CustomerId | Unique customer ID |
| Surname | Customer surname |
| CreditScore | Credit score of the customer |
| Geography | Customer country |
| Gender | Male/Female |
| Age | Customer age |
| Tenure | Number of years the customer has been with the bank |
| Balance | Account balance |
| NumOfProducts | Number of bank products the customer uses |
| HasCrCard | Has credit card (0/1) |
| IsActiveMember | Is active member (0/1) |
| EstimatedSalary | Estimated annual salary |
| Exited | Churn indicator (0 = No, 1 = Yes) |

---

## Features

1. **Data Preprocessing**:
   - Dropped irrelevant columns (`RowNumber`, `CustomerId`, `Surname`).  
   - Encoded `Gender` using LabelEncoder.  
   - One-hot encoded `Geography`.  
   - Standardized numerical features using `StandardScaler`.  

2. **Model**:
   - Artificial Neural Network (ANN) built with **TensorFlow/Keras**:
     - Input layer: 12 neurons (matching the number of features).  
     - Hidden layers: 64 and 32 neurons with ReLU activation.  
     - Output layer: 1 neuron with Sigmoid activation.  
   - Optimizer: Adam  
   - Loss function: Binary Crossentropy  
   - Metrics: Accuracy  

3. **Deployment**:
   - Interactive web interface using **Streamlit**.  
   - Users can select customer information and get churn predictions with probability.  

---

## Installation

1. Clone this repository:

```bash
git clone <repository_url>
cd <repository_folder>
```
2. Install Dependencies
   
```bash
pip install -r requirements.txt
```

---

## Project Structure

```
Customer-Churn-Prediction/
│
├── experiments.ipynb # Data preprocessing & model training
├── predictions.ipynb # Notebook to test predictions
├── app.py # Streamlit app
├── Churn_Modelling.csv # Dataset
├── model.h5 # Trained ANN model
├── label_encoder_gender.pkl # Pickle file for Gender encoder
├── onehot_encoder_geo.pkl # Pickle file for Geography encoder
├── scalar.pkl # Pickle file for scaler
├── requirements.txt # Python dependencies
└── README.md # Project documentation
```
