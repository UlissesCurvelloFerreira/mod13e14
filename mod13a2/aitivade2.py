import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def gerar_grafico_temperatura():
    """Gera um gráfico de linha da evolução da temperatura durante o dia."""

    # Dados fictícios de temperatura para cada hora
    horas = np.arange(0, 25)  # Horários de 0 a 24
    temperaturas = np.array([
        15, 16, 17, 18, 19, 20, 22, 24, 26, 28, 29, 30, 30,
        29, 28, 26, 24, 22, 21, 20, 19, 18, 17, 16, 15
    ])
    
    # Criando um DataFrame com as horas e temperaturas
    df = pd.DataFrame({
        'Hora': horas,
        'Temperatura (°C)': temperaturas
    })

    # Criação do gráfico
    plt.figure(figsize=(12, 6))
    plt.plot(df['Hora'], df['Temperatura (°C)'], marker='o', linestyle='-', color='b')

    # Personalizando o gráfico
    plt.title('Evolução da Temperatura Durante o Dia', fontsize=16)
    plt.xlabel('Hora do Dia', fontsize=12)
    plt.ylabel('Temperatura (°C)', fontsize=12)
    plt.xticks(df['Hora'])              # Definir os ticks no eixo X para cada hora
    plt.yticks(np.arange(15, 36, 1))    # Definir os ticks do eixo Y de 15 a 35, com intervalo de 1
    plt.grid(True)

    # Exibe o gráfico
    plt.show()

if __name__ == "__main__":
    gerar_grafico_temperatura()
