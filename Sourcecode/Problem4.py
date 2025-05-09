import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


df = pd.read_csv("results.csv")


df = df[pd.to_numeric(df['stats_standard_Min'], errors='coerce') > 900]



import numpy as np
np.random.seed(42)
df['TransferValue(EUR)'] = np.random.uniform(1_000_000, 80_000_000, size=len(df)).round(-5)

print(f"Mock transfer values assigned to {len(df)} players.")


features = [
    'stats_standard_Gls', 'stats_standard_Ast',
    'stats_standard_xG', 'stats_standard_xAG',
    'stats_passing_Cmp%', 'stats_shooting_SoT%',
    'stats_defense_Tkl', 'stats_misc_Fld'
]


df_model = df[features + ['TransferValue(EUR)']].copy()
df_model = df_model.dropna()


X = df_model[features]
y = df_model['TransferValue(EUR)']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f" Model trained using Linear Regression.")
print(f" RMSE on test set: €{rmse:,.0f}")


print("\n Feature importances:")
for feature, coef in zip(features, model.coef_):
    print(f"{feature:<30}: €{coef:,.0f} per unit")


df[['Name', 'TransferValue(EUR)']].to_csv("player_transfer_values.csv", index=False)
print(" File saved: player_transfer_values.csv")
