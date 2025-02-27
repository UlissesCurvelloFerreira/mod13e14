import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Gerando dados fictícios (horas de exercício x risco de doença cardíaca)
data = {
    'horas_exercicio': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'risco_doenca': [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
}

# Convertendo para DataFrame
df = pd.DataFrame(data)

# Dividindo as variáveis independentes (X) e dependente (y)
X = df[['horas_exercicio']]
y = df['risco_doenca']

# Dividindo os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criando o modelo de Regressão Logística
model = LogisticRegression()

# Treinando o modelo
model.fit(X_train, y_train)

# Fazendo previsões com os dados de teste
y_pred = model.predict(X_test)

# Avaliando o modelo
accuracy = accuracy_score(y_test, y_pred)

# Exibindo os resultados
print(f"Acurácia do modelo: {accuracy:.2f}")
print("Previsões para todos os possíveis valores:")
for horas in range(1, 11):
    previsao = model.predict(pd.DataFrame([[horas]], columns=['horas_exercicio']))[0]
    print(f"Horas de exercício: {horas} -> Risco de doença: {'Sim' if previsao == 1 else 'Não'}")
