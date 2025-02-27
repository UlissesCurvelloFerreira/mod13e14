import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# URL da API CoinGecko para pegar dados históricos do Bitcoin
url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
params = {
    'vs_currency': 'usd',
    'days': '30',  # Pega os dados dos últimos 30 dias
    'interval': 'daily'
}

# Fazendo a requisição GET para a API CoinGecko
response = requests.get(url, params=params)

# Verifica se a requisição foi bem-sucedida
if response.status_code == 200:
    data = response.json()
    print("Dados recebidos:")
    print(data['prices'])  # Exibe os preços recebidos
else:
    print(f"Erro: {response.status_code}")
    exit()

# Transformando os dados de preços em um DataFrame pandas
prices = data['prices']
df = pd.DataFrame(prices, columns=['timestamp', 'price'])

# Convertendo o timestamp para data
df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
df.set_index('date', inplace=True)

# Calculando a variação de preço (para prever se o preço aumentará ou diminuirá)
df['price_change'] = df['price'].pct_change()  # Percentual de variação entre os dias
df['target'] = (df['price_change'] > 0).astype(int)  # 1 se o preço aumentou, 0 se diminuiu

# Removendo o primeiro valor, pois o cálculo de variação depende do dia anterior
df.dropna(inplace=True)

# Preparando os dados para o modelo
X = df[['price']]  # Usando apenas o preço como variável de entrada
y = df['target']  # O target é se o preço vai aumentar ou não

# Dividindo os dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criando o pipeline
pipeline = make_pipeline(
    StandardScaler(),  # Normalizando os dados
    RandomForestClassifier(n_estimators=100, random_state=42)  # Modelo RandomForest
)

# Treinando o modelo
pipeline.fit(X_train, y_train)

# Fazendo previsões
y_pred = pipeline.predict(X_test)

# Avaliando a acurácia
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia do modelo: {accuracy * 100:.2f}%")

# Visualizando a importância das variáveis (no caso, o preço)
model = pipeline.named_steps['randomforestclassifier']
importances = model.feature_importances_

# Plotando a importância das variáveis
plt.bar(X.columns, importances)
plt.title('Importância das Variáveis no Modelo')
plt.xlabel('Variáveis')
plt.ylabel('Importância')
plt.show()

# Salvando os dados históricos em um arquivo CSV
df.to_csv('crypto_data.csv', index=True)
print("\nDados salvos no arquivo 'crypto_data.csv'.")
