import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Criando os vetores de notas para Matemática e Ciências (sem agrupamento explícito)
matematica = [95, 92, 88, 91, 89, 93, 94, 90, 96, 92, 
              97, 98, 85, 87, 91, 94, 93, 96, 99, 95,
              80, 75, 78, 82, 70, 76, 74, 80, 79, 71, 
              77, 83, 75, 78, 81, 79, 73, 72, 80, 77,
              45, 50, 40, 55, 52, 48, 47, 53, 49, 42, 
              51, 46, 40, 43, 38, 41, 44, 39, 50, 47]

ciencias = [98, 94, 91, 93, 90, 89, 92, 95, 97, 93, 
            99, 98, 86, 90, 92, 91, 93, 98, 94, 92,
            78, 74, 72, 79, 75, 80, 74, 76, 72, 78, 
            79, 80, 77, 73, 74, 79, 76, 75, 71, 77,
            50, 45, 42, 48, 41, 49, 47, 45, 43, 44, 
            42, 46, 43, 40, 44, 48, 45, 42, 41, 49]

# Criando o DataFrame com as duas listas
dados_alunos = pd.DataFrame({
    'Matematica': matematica,
    'Ciencias': ciencias
})

# Aplicando K-Means para agrupar os dados em 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(dados_alunos)  # Ajusta o modelo com os dados
dados_alunos['Cluster'] = kmeans.labels_ # Adicionando os rótulos de cluster aos dados

plt.figure(figsize=(8, 6))  # Visualizando os clusters
scatter = plt.scatter(dados_alunos['Matematica'], dados_alunos['Ciencias'], c=dados_alunos['Cluster'], cmap='coolwarm')

# Adicionando título e rótulos aos eixos
plt.xlabel('Notas de Matemática')
plt.ylabel('Notas de Ciências')
plt.title('Clusters de Alunos com Base nas Notas')

# Ativando o grid no gráfico
plt.grid(True)
plt.colorbar(scatter)  # Barra de cores para visualizar a correspondência entre cores e clusters
plt.show()
