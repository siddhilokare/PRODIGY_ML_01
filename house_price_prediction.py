import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
train_df = pd.read_csv("data/train.csv")

features = ['GrLivArea', 'BedroomAbvGr', 'FullBath']
target = 'SalePrice'


train_df = train_df[features + [target]].dropna()

# Split data (80% train, 20% validation)
X = train_df[features]
y = train_df[target]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)


y_val_pred = model.predict(X_val)

# Evaluate performance
mse = mean_squared_error(y_val, y_val_pred)
r2 = r2_score(y_val, y_val_pred)

print(f"Mean Squared Error (MSE) on Validation Set: {mse:.2f}")
print(f"R² Score on Validation Set: {r2:.4f}")


train_predictions = model.predict(X)

train_results = pd.DataFrame({
    'Id': train_df.index,
    'Actual_SalePrice': y,
    'Predicted_SalePrice': train_predictions
})


train_results.to_csv("data/train_predictions.csv", index=False)
print("Predictions saved in train_predictions.csv")


plt.scatter(y, train_predictions, alpha=0.5)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices (Train.csv)")
plt.show()
