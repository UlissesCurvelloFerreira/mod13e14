import pandas as pd
#Ulisses Curvello Ferreira

# Caminho do arquivo (ajuste conforme necessário)
caminho_arquivo = "planinha/dados.xlsx"  # Se for CSV, use "planinha/dados.csv"

def carregar_planilha():
    """Carrega a planilha e retorna um DataFrame."""
    return pd.read_excel(caminho_arquivo)  # Se for CSV, use pd.read_csv()

def imprimir_planilha(df):
    """Imprime a planilha de forma organizada."""
    print(df)

def substituir_precipitacao_por_media(df):
    """Substitui valores ausentes na coluna Precipitação pela média da coluna."""
    media_precipitacao = df["Precipitação (mm)"].mean()
    df = df.copy()  # Evita problemas com cópias no Pandas
    df["Precipitação (mm)"] = df["Precipitação (mm)"].fillna(media_precipitacao)
    return df

def substituir_umidade_por_mediana(df):
    """Substitui valores ausentes na coluna 'Umidade Relativa (%)' pela mediana da coluna."""
    mediana_umidade = df["Umidade Relativa (%)"].median()
    df = df.copy()  # Evita problemas com cópias no Pandas
    df["Umidade Relativa (%)"] = df["Umidade Relativa (%)"].fillna(mediana_umidade)
    return df


def adicionar_amplitude_termica(df):
    """Adiciona a coluna Amplitude Térmica, que é a diferença entre Temperatura Máxima e Temperatura Mínima."""
    df = df.copy()  # Evita problemas com cópias no Pandas
    df["Amplitude Térmica"] = df["Temperatura Máxima (°C)"] - df["Temperatura Mínima (°C)"]
    return df

def filtrar_cidades_acima_30(df):
    """Cria um novo DataFrame contendo apenas as cidades com Temperatura Máxima acima de 30°C."""
    df_filtrado = df[df["Temperatura Máxima (°C)"] > 30]
    return df_filtrado

def reordenar_colunas(df):
    """Reordena as colunas do DataFrame para a ordem desejada."""
    colunas_ordenadas = [
        "Data", "Cidade", "Temperatura Máxima (°C)", "Temperatura Mínima (°C)",
        "Amplitude Térmica", "Precipitação (mm)", "Umidade Relativa (%)"
    ]
    df = df[colunas_ordenadas]  # Reordena as colunas
    return df


# Fluxo principal do programa

if __name__ == "__main__":
    df = carregar_planilha()                    # Carregar a planilha

    df = substituir_precipitacao_por_media(df) 
    print("\nAplica a substituição de NaN na Precipitação pela média\n")
    imprimir_planilha(df)
    print('*' * 140)

    df = substituir_umidade_por_mediana(df)
    print("\nSubstitui NaN na Umidade Relativa pela mediana\n")
    imprimir_planilha(df)
    print('*' * 140)

    df = adicionar_amplitude_termica(df)
    print("\nAdiciona a coluna Amplitude Térmica\n")
    imprimir_planilha(df)
    print('*' * 140)

    df_acima_30 = filtrar_cidades_acima_30(df)
    print("\nTemperatura Máxima acima de 30°C\n")
    imprimir_planilha(df_acima_30)
    print('*' * 140)

    df = reordenar_colunas(df)
    print("\nReordena as colunas\n")
    imprimir_planilha(df)
    print('*' * 140)

    # Salvar as alterações
    df.to_excel(caminho_arquivo, index=False)
    df_acima_30.to_excel("planinha/new_dados.xlsx", index=False)

    print("\n\n\nValores ausentes substituídos, planilha atualizada e colunas reordenadas com sucesso!")
