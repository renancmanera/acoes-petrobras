import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("Bovespa.csv")

cols_to_convert = ["Open", "High", "Low", "Close", "Volume", "Adj Close"]
for col in cols_to_convert:
    df[col] = df[col].astype(str).str.replace(",", ".").str.replace(" ", "")
    df[col] = pd.to_numeric(df[col], errors='coerce')

df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y")

df_petrol = df[df["Ticker"].isin(["PETR3", "PETR4"])].copy()
df_petrol.sort_values(by=["Ticker", "Date"], inplace=True)

df_petrol["Retorno"] = df_petrol.groupby("Ticker")["Close"].pct_change(fill_method=None)
df_petrol["MediaMovel_5"] = df_petrol.groupby("Ticker")["Close"].rolling(window=5).mean().reset_index(level=0, drop=True)
df_petrol["Volatilidade_5"] = df_petrol.groupby("Ticker")["Close"].rolling(window=5).std().reset_index(level=0, drop=True)

df_petrol.dropna(subset=["Retorno", "MediaMovel_5", "Volatilidade_5"], inplace=True)

X = df_petrol[["Open", "High", "Low", "Volume", "MediaMovel_5", "Volatilidade_5"]]
y = df_petrol["Close"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

modelo_lr = LinearRegression()
modelo_lr.fit(X_train, y_train)
y_pred_lr = modelo_lr.predict(X_test)

modelo_tree = DecisionTreeRegressor(max_depth=5, random_state=42)
modelo_tree.fit(X_train, y_train)
y_pred_tree = modelo_tree.predict(X_test)

rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
r2_lr = r2_score(y_test, y_pred_lr)

rmse_tree = np.sqrt(mean_squared_error(y_test, y_pred_tree))
r2_tree = r2_score(y_test, y_pred_tree)

print("=== Regressão Linear ===")
print(f"RMSE: {rmse_lr:.4f}")
print(f"R²:   {r2_lr:.4f}\n")

print("=== Árvore de Decisão ===")
print(f"RMSE: {rmse_tree:.4f}")
print(f"R²:   {r2_tree:.4f}\n")

plt.figure(figsize=(12,6))
plt.plot(y_test.values, label="Real", marker='o')
plt.plot(y_pred_lr, label="Linear", marker='x')
plt.plot(y_pred_tree, label="Árvore", marker='^')
plt.title("Previsão do Preço de Fechamento - Linear vs Árvore de Decisão")
plt.xlabel("Tempo")
plt.ylabel("Preço (R$)")
plt.legend()
plt.grid(True)
plt.show()

print("\n=== Estrutura da Árvore de Decisão ===")
print(export_text(modelo_tree, feature_names=list(X.columns)))
