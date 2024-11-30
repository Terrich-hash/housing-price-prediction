# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Simulating a small dataset
data = {
    "Area (sq ft)": [800, 1200, 1500, 1800, 2200, 2500, 3000, 3500],
    "Bedrooms": [2, 3, 3, 4, 4, 5, 5, 6],
    "Price": [150000, 200000, 250000, 300000, 350000, 400000, 450000, 500000]
}
df = pd.DataFrame(data)

# Function: Perform EDA
def perform_eda(data):
    print("\n--- Exploratory Data Analysis ---")
    print(data.describe())
    print("\nCorrelation Matrix:")
    print(data.corr())
    sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.show()

    # Scatter plots
    plt.figure(figsize=(12, 5))
    # Area vs Price
    plt.subplot(1, 2, 1)
    sns.scatterplot(x="Area (sq ft)", y="Price", data=data, color="blue")
    plt.title("Area vs Price")
    plt.xlabel("Area (sq ft)")
    plt.ylabel("Price ($)")
    # Bedrooms vs Price
    plt.subplot(1, 2, 2)
    sns.scatterplot(x="Bedrooms", y="Price", data=data, color="green")
    plt.title("Bedrooms vs Price")
    plt.xlabel("Number of Bedrooms")
    plt.ylabel("Price ($)")
    plt.tight_layout()
    plt.show()

# Function: Train and evaluate a model
def train_and_evaluate(data):
    X = data[["Area (sq ft)", "Bedrooms"]]
    y = data["Price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n--- Model Evaluation ---")
    print("Model Coefficients:", model.coef_)
    print("Model Intercept:", model.intercept_)
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"R^2 Score: {r2:.2f}")

    # Visualizing predictions vs. actuals
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, color='purple', alpha=0.6)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', color='red')
    plt.title("Actual vs Predicted Prices")
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.grid()
    plt.show()

# Function: Make a prediction
def predict_price(area, bedrooms, model):
    prediction = model.predict([[area, bedrooms]])
    print(f"Predicted Price for {area} sq ft and {bedrooms} bedrooms: ${prediction[0]:,.2f}")
    return prediction[0]

# Perform EDA
perform_eda(df)

# Train and evaluate the model
X = df[["Area (sq ft)", "Bedrooms"]]
y = df["Price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the model and make predictions
model = LinearRegression()
model.fit(X_train, y_train)
train_and_evaluate(df)

# Example prediction
example_area = 2000
example_bedrooms = 4
predict_price(example_area, example_bedrooms, model)
