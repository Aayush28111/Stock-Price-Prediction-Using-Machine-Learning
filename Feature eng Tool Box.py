
# Feature Engineering Toolbox

import numpy as np
import pandas as pd

np.random.seed(42)

data = np.random.randn(100, 4) * 10 + 50  # 100 rows, 4 features
df = pd.DataFrame(data, columns=['Feature1', 'Feature2', 'Feature3', 'Feature4'])

df.loc[np.random.randint(0, 100, 5), 'Feature2'] = np.nan
df.loc[np.random.randint(0, 100, 5), 'Feature3'] = np.nan
df.loc[0, 'Feature1'] = 300  
df.loc[1, 'Feature4'] = -200  

print("🔹 Raw Dataset (with missing values and outliers):")
print(df.head())

df_filled = df.copy()
df_filled = df_filled.fillna(df_filled.mean())

def remove_outliers_zscore(df, threshold=3):
    z_scores = (df - df.mean()) / df.std()
    mask = (np.abs(z_scores) < threshold).all(axis=1)
    return df[mask]

df_no_outliers = remove_outliers_zscore(df_filled)

def min_max_scaler(df):
    return (df - df.min()) / (df.max() - df.min())

df_normalized = min_max_scaler(df_no_outliers)

def standard_scaler(df):
    return (df - df.mean()) / df.std()

df_standardized = standard_scaler(df_no_outliers)

X_normalized = df_normalized.to_numpy()
X_standardized = df_standardized.to_numpy()

np.save("X_normalized.npy", X_normalized)
np.save("X_standardized.npy", X_standardized)

print("\n✅ Final Normalized Data Sample:")
print(df_normalized.head())

print("\n✅ Final Standardized Data Sample:")
print(df_standardized.head())

print("\n📦 Shapes:")
print("Normalized:", X_normalized.shape)
print("Standardized:", X_standardized.shape)
