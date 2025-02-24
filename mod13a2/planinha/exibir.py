import pandas as pd
import os

# Caminho do arquivo
caminho_arquivo = os.path.join("planinha", "dados.xlsx")

# Verificando se o arquivo existe
if os.path.exists(caminho_arquivo):
    # Lendo o arquivo Excel
    dados = pd.read_excel(caminho_arquivo)
    
    # Exibindo os dados
    print(dados)
else:
    print(f"O arquivo {caminho_arquivo} não foi encontrado.")
