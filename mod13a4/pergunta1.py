import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Criando dados fictícios
dados = {
    "Combustivel": ["Gasolina", "Diesel", "Etanol", "Gasolina", "Diesel", "Etanol", "Gasolina", "Diesel", "Etanol", "Gasolina"],
    "Idade": [5, 3, 8, 2, 7, 1, 4, 6, 9, 2],
    "Quilometragem": [50000, 30000, 80000, 20000, 70000, 10000, 40000, 60000, 90000, 25000],
    "Preco": [30000, 35000, 20000, 40000, 22000, 45000, 32000, 25000, 18000, 42000]
}

df = pd.DataFrame(dados)

# Separando variáveis independentes e dependente
X = df.drop(columns=["Preco"])
y = df["Preco"]

# Definição das colunas categóricas e numéricas
colunas_categoricas = ["Combustivel"]
colunas_numericas = ["Idade", "Quilometragem"]

# Criando transformações (pipeline sendo usado)
transformador_categorico = OneHotEncoder(handle_unknown="ignore")
transformador_numerico = StandardScaler()

# Aplicando transformações às colunas apropriadas
preprocessador = ColumnTransformer(
    transformers=[
        ("cat", transformador_categorico, colunas_categoricas),
        ("num", transformador_numerico, colunas_numericas)
    ]
)

# Criando o pipeline
pipeline = Pipeline([
    ("preprocessador", preprocessador),
    ("modelo", LinearRegression())
])

# Dividindo os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinando o modelo
pipeline.fit(X_train, y_train)

# Fazendo previsões
y_pred = pipeline.predict(X_test)

# Avaliando o modelo
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)  # Raiz quadrada do MSE

print(f"Erro Quadrático Médio (MSE): {mse:.2f}")
print(f"Erro Médio Quadrático Médio (RMSE): {rmse:.2f}")


# Comparação entre valores reais e previstos
print("\nComparação entre valores reais e previstos:")
previsoes_treino = pipeline.predict(X)

for combustivel, idade, km, preco_real, preco_previsto in zip(df["Combustivel"], df["Idade"], df["Quilometragem"], df["Preco"], previsoes_treino):
    print(f"Carro {combustivel}, {idade} anos, {km} km → Preço real: R$ {preco_real:.2f} | Preço previsto: R$ {preco_previsto:.2f}")

# Testando previsões com novos dados
novos_dados = pd.DataFrame({
    "Combustivel": ["Gasolina", "Diesel", "Etanol"],
    "Idade": [4, 6, 2],
    "Quilometragem": [45000, 70000, 15000]
})

previsoes_novos = pipeline.predict(novos_dados)

print("\nPrevisões para novos carros:")
for combustivel, idade, km, preco in zip(novos_dados["Combustivel"], novos_dados["Idade"], novos_dados["Quilometragem"], previsoes_novos):
    print(f"Carro {combustivel}, {idade} anos, {km} km → Preço previsto: R$ {preco:.2f}")
