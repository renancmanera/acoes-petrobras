# importando bibliotecas
# pandas para manipular dados em formato de tabela, df = dataframe
# matplotlib para plotar gráficos
# sklearn para criar modelos de regressão linear e random forest

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# carrega os dados e transforma para o formato correto
df = pd.read_csv('Bovespa.csv', parse_dates=['Date'], dayfirst=True)

# converte formatos numéricos (pt-br -> float)
for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
    df[col] = (df[col]
               .astype(str)
               .str.replace('.', '', regex=False)
               .str.replace(',', '.', regex=False))
    df[col] = pd.to_numeric(df[col], errors='coerce')

# função para processar cada ticker (PETR3 e PTR4)
# filtra linhas do ticker escolhido
# ordena por data e reseta o índice
def processar_ticker(ticker_nome):
    df_t = df[df['Ticker'] == ticker_nome].sort_values('Date').reset_index(drop=True)

    # criando uma nova coluna lag de fechamento do dia anterior
    # shift(1) desloca os dados para baixo, para ter um valor null no primeiro dia
    # dropna() remove linhas com valores ausentes
    # reset_index(drop=True) redefine o índice do dataframe
    df_t['Close_lag1'] = df_t['Close'].shift(1)
    df_t = df_t.dropna().reset_index(drop=True)

    # variaveis de entrada
    feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Close_lag1']

    # minimo de dias para o treinamento do modelo
    min_train = 30

    # criando listas para armazenar os dados a serem plotados
    plot_dates = [] # datas dos pontos previstos
    real_close = [] # preços reais de fechamento
    lin_preds = [] # previsões do modelo de regressão linear
    rf_preds = [] # previsões do modelo de random forest

    # loop para prever o preço de fechamento para cada dia a partir do dia min_train
    for i in range(min_train, len(df_t)):
        X_train = df_t[feature_cols].iloc[:i] # dados de treinamento
        y_train = df_t['Close'].iloc[1:i+1] # preços reais de fechamento
        X_pred  = df_t[feature_cols].iloc[i:i+1] # dados para previsão

        lr = LinearRegression() # cria o modelo de regressão linear
        lr.fit(X_train, y_train)
        rf = RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_leaf=5, random_state=42) 
        # cria o modelo de random forest
        rf.fit(X_train, y_train)

        # faz a previsão para o dia atual (i)
        lin_preds.append(lr.predict(X_pred)[0])
        rf_preds.append(rf.predict(X_pred)[0])

        # adiciona a data e o preço real de fechamento para o dia atual (i)
        plot_dates.append(df_t['Date'].iloc[i])
        real_close.append(df_t['Close'].iloc[i])

    plt.figure(figsize=(10, 5))
    plt.plot(plot_dates, real_close, label='Real Close')
    plt.plot(plot_dates, lin_preds, label='Linear Regression', linewidth=1)
    plt.plot(plot_dates, rf_preds, label='Random Forest', linewidth=1)
    plt.title(f'{ticker_nome}: Walk-Forward com Janela Mínima de {min_train} Dias')
    plt.xlabel('Data')
    plt.ylabel('Preço de Fechamento (R$)')
    plt.legend()
    plt.tight_layout()
    plt.show()


# processar e plotar PETR3
processar_ticker('PETR3')

# processar e plotar PTR4
processar_ticker('PTR4')