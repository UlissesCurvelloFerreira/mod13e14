import requests
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Função para coletar dados históricos de preços de criptomoedas da CoinGecko API
def coletar_dados(moeda_id='bitcoin', periodo='365', vs_currency='usd'):
    url = f'https://api.coingecko.com/api/v3/coins/{moeda_id}/market_chart?vs_currency={vs_currency}&days={periodo}'
    resposta = requests.get(url)
    
    if resposta.status_code == 200:
        dados = resposta.json()
        return pd.DataFrame(dados['prices'], columns=['timestamp', 'price'])
    else:
        print(f"Erro na requisição. Status code: {resposta.status_code}")
        return None

# Função para processar os dados
def processar_dados(dados):
    # Convertendo timestamp para formato de data
    dados['date'] = pd.to_datetime(dados['timestamp'], unit='ms')
    dados.set_index('date', inplace=True)
    dados.drop(columns=['timestamp'], inplace=True)
    
    # Calculando a variação percentual do preço
    dados['price_change'] = dados['price'].pct_change()
    
    # Calculando médias móveis (exemplo de engenharia de características)
    dados['moving_average_7'] = dados['price'].rolling(window=7).mean()
    dados['moving_average_30'] = dados['price'].rolling(window=30).mean()
    
    # Indicadores técnicos simples, como desvio padrão (volatilidade)
    dados['price_volatility'] = dados['price'].rolling(window=30).std()
    
    # Target: 1 se o preço vai subir, 0 se cair
    dados['target'] = (dados['price_change'] > 0).astype(int)
    dados.dropna(inplace=True)  # Remover valores nulos (resultantes das médias móveis e desvios)
    
    return dados

# Função para treinar e avaliar o modelo
def treinar_modelo(dados):
    X = dados[['price', 'moving_average_7', 'moving_average_30', 'price_volatility']]  # Mais features
    y = dados['target']  # O target é se o preço vai aumentar ou não
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Criando um pipeline com pré-processamento e RandomForest
    pipeline = make_pipeline(
        StandardScaler(),
        RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)  # Ajustando o modelo
    )
    
    pipeline.fit(X_train, y_train)  # Treinando o modelo
    y_pred = pipeline.predict(X_test)  # Fazendo previsões
    
    accuracy = accuracy_score(y_test, y_pred)  # Avaliando a acurácia
    print(f"Acurácia do modelo: {accuracy * 100:.2f}%")
    
    # Validação cruzada para melhorar a confiabilidade da acurácia
    cv_scores = cross_val_score(pipeline, X, y, cv=5)
    print(f"Acurácia média da validação cruzada: {cv_scores.mean() * 100:.2f}%")
    
    # Exibindo a importância das variáveis
    model = pipeline.named_steps['randomforestclassifier']
    importances = model.feature_importances_
    
    plt.bar(X.columns, importances)
    plt.title('Importância das Variáveis no Modelo')
    plt.xlabel('Variáveis')
    plt.ylabel('Importância')
    plt.show()

# Função para salvar os dados em CSV
def salvar_dados(dados, nome_arquivo='crypto_data.csv'):
    dados.to_csv(nome_arquivo, index=True)
    print(f"\nDados salvos no arquivo '{nome_arquivo}'.")

# Função para gerar gráfico linear do preço da criptomoeda
def gerar_grafico_linear(dados):
    # Plotando o gráfico linear do preço ao longo do tempo
    plt.figure(figsize=(10, 6))
    plt.plot(dados.index, dados['price'], label='Preço de Fechamento', color='blue')
    plt.title('Preço da Criptomoeda ao Longo do Tempo')
    plt.xlabel('Data')
    plt.ylabel('Preço (USD)')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()  # Ajusta o layout para exibir o gráfico sem cortar partes
    plt.legend()
    plt.show()

# Função principal
def main():
    moeda_id = 'bitcoin'  # ID da moeda no CoinGecko
    periodo = '365'  # Dados para o último ano (365 dias)
    
    print(f"Coletando dados históricos para {moeda_id}...")
    dados = coletar_dados(moeda_id, periodo)
    
    if dados is not None:
        print(f"Processando dados...")
        dados_processados = processar_dados(dados)
        
        # Salvando os dados em CSV
        salvar_dados(dados_processados)
        
        # Gerando gráfico linear
        gerar_grafico_linear(dados_processados)
        
        # Treinando o modelo e avaliando
        treinar_modelo(dados_processados)
    
# Chamando a função principal
if __name__ == "__main__":
    main()
