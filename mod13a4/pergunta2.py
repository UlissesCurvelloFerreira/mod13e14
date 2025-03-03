import requests                                         # Requisições HTTP;
import pandas as pd                                     # Manipulação e análise de dados;
from sklearn.model_selection import train_test_split    # Função para dividir os dados em conjuntos de treinamento e teste;
from sklearn.ensemble import RandomForestClassifier     # Algoritmo de aprendizado de máquina (Random Forest) para classificação;
from sklearn.pipeline import make_pipeline              # Utilizada para criar um pipeline de pré-processamento e modelagem;
from sklearn.preprocessing import StandardScaler        # Função para normalizar os dados;
from sklearn.metrics import accuracy_score              # Função para calcular a acurácia do modelo;
import matplotlib.pyplot as plt                         # Biblioteca para criar gráficos e visualizações;


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
else:
    print(f"Erro: {response.status_code}")
    exit()

# Transformando os dados de preços em um DataFrame pandas
prices = data['prices']
volume = data['total_volumes']  # Obtendo o volume de transações
df = pd.DataFrame(prices, columns=['timestamp', 'price'])
df['volume'] = [v[1] for v in volume]  # Adicionando volume de transações

# Convertendo o timestamp para data
df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
df.set_index('date', inplace=True)

# Calculando a variação de preço (para prever se o preço aumentará ou diminuirá)
df['price_change'] = df['price'].pct_change()  # Percentual de variação entre os dias
df['target'] = (df['price_change'] > 0).astype(int)  # 1 se o preço aumentou, 0 se diminuiu

# Removendo o primeiro valor, pois o cálculo de variação depende do dia anterior
df.dropna(inplace=True)

# Calculando a média móvel de 7 dias
df['ma_7'] = df['price'].rolling(window=7).mean()

# Calculando o RSI de 14 dias
def compute_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

df['rsi'] = compute_rsi(df['price'])

# Preparando os dados para o modelo (agora incluindo as novas variáveis)
X = df[['price', 'ma_7', 'rsi', 'volume']]  # Incluindo preço, média móvel, RSI e volume de transações
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

# Prevendo para o próximo dia (usando o último valor disponível)
ultimo_preco = df['price'].iloc[-1]  # Último preço
ultimo_ma_7 = df['ma_7'].iloc[-1]  # Última média móvel
ultimo_rsi = df['rsi'].iloc[-1]  # Último RSI
ultimo_volume = df['volume'].iloc[-1]  # Último volume

ultimo_dado = pd.DataFrame([[ultimo_preco, ultimo_ma_7, ultimo_rsi, ultimo_volume]], columns=['price', 'ma_7', 'rsi', 'volume'])

# Fazendo a previsão
previsao = pipeline.predict(ultimo_dado)

# Exibindo a previsão para o próximo dia
if previsao == 1:
    print("\033[32m Previsão: O preço do Bitcoin deve SUBIR amanhã. \033[0m")
else:
    print("\033[31m Previsão: O preço do Bitcoin deve DESCER amanhã. \033[0m")

# Visualizando o gráfico do preço do Bitcoin ao longo do tempo
plt.figure(figsize=(10, 5))
plt.plot(df.index, df['price'], label='Preço Bitcoin', color='blue')
plt.title('Preço do Bitcoin nos últimos 30 dias')
plt.xlabel('Data')
plt.ylabel('Preço (USD)')
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Visualizando a importância das variáveis
model = pipeline.named_steps['randomforestclassifier']
importances = model.feature_importances_

# Plotando a importância das variáveis
plt.figure(figsize=(6, 4))
plt.bar(['price', 'ma_7', 'rsi', 'volume'], importances)  # Exibindo a importância de cada variável
plt.title('Importância das Variáveis no Modelo')
plt.xlabel('Variáveis')
plt.ylabel('Importância')
plt.show()

# Salvando os dados históricos em um arquivo CSV
df.to_csv('crypto_data.csv', index=True)
print("\nDados salvos no arquivo 'crypto_data.csv'.")