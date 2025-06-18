# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import yfinance as yf

df = yf.download('AAPL', start='2018-01-01', end='2023-12-31')
df = df[['Close']]  
df.dropna(inplace=True)
df['Prediction'] = df['Close'].shift(-1)  

X = np.array(df[['Close']])[:-1]  
y = np.array(df['Prediction'])[:-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error:", mse)

plt.figure(figsize=(12, 6))
plt.plot(y_test, label='Actual Price')
plt.plot(predictions, label='Predicted Price')
plt.title('Stock Price Prediction (Linear Regression)')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
