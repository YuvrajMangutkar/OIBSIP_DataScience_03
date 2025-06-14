import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

#Load data
df = pd.read_csv("car data.csv")

#Feature engineering
df['Car_Age'] = 2025 - df['Year']
df_cleaned = df.drop(['Car_Name', 'Year'], axis=1)
df_cleaned = pd.get_dummies(df_cleaned, drop_first=True)

#Split features and target
X = df_cleaned.drop('Selling_Price', axis=1)
y = df_cleaned['Selling_Price']

#Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Train models
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

rf_model = RandomForestRegressor(random_state=42, n_estimators=100)
rf_model.fit(X_train, y_train)

#Evaluate models
# Define evaluate function
from math import sqrt
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = sqrt(mean_squared_error(y_test, y_pred))
    return y_pred, {"R2 Score": r2, "MAE": mae, "RMSE": rmse}


# Train models
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

rf_model = RandomForestRegressor(random_state=42, n_estimators=100)
rf_model.fit(X_train, y_train)

# Evaluate models
lr_pred, lr_results = evaluate_model(lr_model, X_test, y_test)
rf_pred, rf_results = evaluate_model(rf_model, X_test, y_test)

# Print results
print("Linear Regression:", lr_results)
print("Random Forest:", rf_results)


#Visualization: Actual vs Predicted (Random Forest)
plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test, y=rf_pred)
plt.xlabel("Actual Selling Price (Lakh)")
plt.ylabel("Predicted Selling Price (Lakh)")
plt.title("Actual vs Predicted Selling Price (Random Forest)")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.show()

#Feature importance
importances = pd.Series(rf_model.feature_importances_, index=X.columns)
importances.sort_values().plot(kind='barh', figsize=(8,6))
plt.title("Feature Importance (Random Forest)")
plt.show()

#input
print("\n📌 Please enter car details to predict the price:")

present_price = float(input("Present Price (in Lakh ₹): "))
kms_driven = int(input("Kilometers Driven: "))
owner = int(input("Number of previous owners (0/1/3): "))
car_age = int(input("Age of the car (Years): "))

fuel_type = input("Fuel Type (Petrol/Diesel/CNG): ").strip().capitalize()
selling_type = input("Selling Type (Dealer/Individual): ").strip().capitalize()
transmission = input("Transmission Type (Manual/Automatic): ").strip().capitalize()

# Prepare input DataFrame
input_data = {
    'Present_Price': [present_price],
    'Driven_kms': [kms_driven],  # match the training feature name exactly!
    'Owner': [owner],
    'Car_Age': [car_age],
    'Fuel_Type_Diesel': [1 if fuel_type == 'Diesel' else 0],
    'Fuel_Type_Petrol': [1 if fuel_type == 'Petrol' else 0],
    'Selling_type_Individual': [1 if selling_type == 'Individual' else 0],
    'Transmission_Manual': [1 if transmission == 'Manual' else 0]
}


input_df = pd.DataFrame(input_data)

# Predict
predicted_price = rf_model.predict(input_df)[0]
print(f"\nPredicted Selling Price: ₹ {predicted_price:.2f} Lakh")


