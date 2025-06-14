# Car Price Prediction — Oasis Infobyte Virtual Internship (Task 3)

This repository contains my implementation of **Car Price Prediction**, developed as part of **Task 3 of the Oasis Infobyte Data Science Virtual Internship Program**.

## **Objective**
The goal of this project is to build a machine learning model that can predict the selling price of a car based on its features, such as:
- Present price
- Kilometers driven
- Owner type
- Age of the car
- Fuel type
- Transmission type
- Seller type  

This project helps in understanding how machine learning can assist in predicting fair prices of used cars.

## **Tools & Libraries Used**
- Python 3.x
- pandas (data manipulation)
- scikit-learn (modeling and evaluation)
- matplotlib / seaborn (optional, for visualization)

## **Steps Performed**

1️⃣ **Data Loading**
- Loaded the dataset (`car data.csv`) into a pandas DataFrame.

2️⃣ **Data Cleaning**
- Checked for and handled missing values (none were found in the dataset).
- Renamed columns for consistency (e.g., `Year` converted to `Car_Age`).
- Converted categorical features using one-hot encoding:
  - Fuel Type
  - Seller Type
  - Transmission

3️⃣ **Feature Engineering**
- Derived `Car_Age` from `Year` column.
- Dropped irrelevant columns like `Car_Name` after encoding necessary info.

4️⃣ **Modeling**
- Split data into training and test sets (80/20 split).
- Trained a **RandomForestRegressor** and **LinearRegression** model.
- Evaluated using R² score and Mean Absolute Error (MAE).

5️⃣ **Prediction**
- Model accepts user input (via code) and predicts the selling price of a car.

## **Outcome**
- The **Random Forest model** achieved better accuracy (higher R² score) than linear regression.
- The model can predict a car's price based on its features and provides reasonable estimates.
