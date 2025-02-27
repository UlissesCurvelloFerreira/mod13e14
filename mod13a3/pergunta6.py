import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data = {
    'num_aparelhos': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'custo_mensal': [50, 60, 90, 100, 130, 140, 170, 180, 210, 220]
}

df = pd.DataFrame(data) # Convertendo para DataFrame

# Dividindo as variáveis independentes (X) e dependente (y)
X = df[['num_aparelhos']]
y = df['custo_mensal']

# Dividindo os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()                  # Criando o modelo de Regressão Linear
model.fit(X_train, y_train)                 # Treinando o modelo
y_pred = model.predict(X_test)              # Fazendo previsões com os dados de teste
mse = mean_squared_error(y_test, y_pred)    # Avaliando o modelo
print(f"Erro Quadrático Médio: {mse}")

# Exibindo o coeficiente angular e o intercepto
a = model.coef_[0] 
b = model.intercept_
print(f"Equação da reta: y = {a:.2f} * x + {b:.2f}")

# Testando com um novo valor
teste_aparelhos = 9
previsao = model.predict(pd.DataFrame([[teste_aparelhos]], columns=['num_aparelhos']))[0]
calculo_manual = a * teste_aparelhos + b

print(f"Para {teste_aparelhos} aparelhos, previsão pelo modelo: R${previsao:.2f}")
print(f"Cálculo manual: y = {a:.2f} * {teste_aparelhos} + {b:.2f} = R${calculo_manual:.2f}")
