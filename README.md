# 📊 Predição de Ações PETR3 e PETR4 com Machine Learning

Este projeto aplica técnicas de Machine Learning para prever os preços de fechamento das ações da Petrobras (PETR3 e PETR4), utilizando dados históricos da B3. Foram desenvolvidos e comparados dois modelos preditivos: Regressão Linear e Árvore de Decisão, com foco na análise estatística e estrutura binária da árvore.

## 🔧 Tecnologias Utilizadas
- Python 3.11+
- pandas
- numpy
- matplotlib
- scikit-learn

## 📁 Estrutura
- `Bovespa.csv`: Base de dados histórica da B3 (2015–2016)
- `main.py`: Código principal com pré-processamento, modelos e visualizações
- `arvore_decisao.png`: Imagem gerada da árvore de decisão
- `README.md`: Descrição do projeto

## 📌 Funcionalidades
- Filtragem de dados PETR3 e PETR4
- Criação de variáveis derivadas (retorno, média móvel, volatilidade)
- Treinamento e avaliação de Regressão Linear e DecisionTreeRegressor
- Visualização gráfica de previsões e estrutura da árvore
- Cálculo de RMSE e R² para comparação de modelos

## 📈 Resultados
- A Regressão Linear apresentou melhor desempenho preditivo (R² > 0.97)
- A Árvore de Decisão permitiu explorar logicamente as divisões entre variáveis
