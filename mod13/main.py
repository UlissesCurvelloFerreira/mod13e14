import matplotlib.pyplot as plt

# Dados para o gráfico
x = [1, 2, 3, 4, 5]
y = [1, 4, 9, 16, 25]

# Criação do gráfico
plt.plot(x, y, marker='o', linestyle='-', color='b', label='x²')

# Personalização
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Gráfico de Linha - Função Quadrática')
plt.legend()

# Exibição do gráfico
plt.show()


#pip install matplotlib
#pip uninstall matplotlib
