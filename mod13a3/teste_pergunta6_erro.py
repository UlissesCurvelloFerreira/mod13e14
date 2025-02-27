import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Gerando dados fictícios (número de aparelhos x custo mensal em R$)
data = {
    'num_aparelhos': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'custo_mensal': [50, 60, 90, 100, 130, 140, 170, 180, 210, 220]
}

# Convertendo para DataFrame
df = pd.DataFrame(data)

# Dividindo as variáveis independentes (X) e dependente (y)
X = df[['num_aparelhos']]
y = df['custo_mensal']

# Dividindo os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criando o modelo de Regressão Linear
model = LinearRegression()

# Treinando o modelo
model.fit(X_train, y_train)

# Fazendo previsões com os dados de teste
y_pred = model.predict(X_test)

# Avaliando o modelo
mse = mean_squared_error(y_test, y_pred)
print(f"Erro Quadrático Médio: {mse}")

# Exibindo o coeficiente angular e o intercepto
a = model.coef_[0]  # Inclinação da reta
b = model.intercept_  # Intercepto
print(f"Equação da reta: y = {a:.2f}x + {b:.2f}")

# Função para prever custo com erro estimado
def prever_custo(num_aparelhos, modelo, coef, inter, erro):
    previsao = modelo.predict(pd.DataFrame([[num_aparelhos]], columns=['num_aparelhos']))[0]
    calculo_manual = coef * num_aparelhos + inter
    erro_estimado = np.sqrt(erro)  # Raiz quadrada do MSE como estimativa do erro padrão
    print(f"Para {num_aparelhos} aparelhos:")
    print(f"Previsão pelo modelo: R${previsao:.2f} ± R${erro_estimado:.2f}")
    print(f"Cálculo manual: y = {coef:.2f} * {num_aparelhos} + {inter:.2f} = R${calculo_manual:.2f}")

# Testando com um novo valor
teste_aparelhos = 6
prever_custo(teste_aparelhos, model, a, b, mse)