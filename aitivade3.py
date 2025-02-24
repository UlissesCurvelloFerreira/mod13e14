import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Simulando dados para uma loja durante uma semana com padrões realistas
dados = {
    'Dia': ['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado', 'Domingo'],
    'Vendas': [1500, 1800, 1200, 2000, 2500, 3000, 1700],  # Total de vendas por dia em reais
    'Clientes': [50, 55, 60, 65, 70, 75, 80],  # Número de clientes atendidos por dia
    'Lucro': [400, 350, 250, 400, 550, 600, 390]  # Lucro gerado em reais por dia
}

# Criando o DataFrame
df = pd.DataFrame(dados)

# Gráfico 1: Gráfico de barras para o total de vendas por dia
plt.figure(figsize=(10, 6))
sns.barplot(x='Dia', y='Vendas', data=df, hue='Dia', palette='viridis', legend=False)  # Atribuindo 'Dia' ao 'hue'
plt.title('Total de Vendas por Dia', fontsize=16)
plt.xlabel('Dia da Semana', fontsize=12)
plt.ylabel('Total de Vendas (R$)', fontsize=12)
plt.show()

# Gráfico 2: Gráfico de dispersão relacionando número de clientes e total de vendas
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Clientes', y='Vendas', data=df, color='blue', s=100, edgecolor='black')
plt.title('Relação entre Número de Clientes e Total de Vendas', fontsize=16)
plt.xlabel('Número de Clientes', fontsize=12)
plt.ylabel('Total de Vendas (R$)', fontsize=12)
plt.show()

# Gráfico 3: Heatmap mostrando a correlação entre as variáveis de Vendas, Clientes e Lucro
correlacao = df[['Vendas', 'Clientes', 'Lucro']].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlacao, annot=True, cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5)
plt.title('Correlação entre Vendas, Clientes e Lucro', fontsize=16)
plt.show()
