import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Criação de uma figura
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Dados para o gráfico 3D
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = np.sin(np.sqrt(X**2 + Y**2))

# Criação da superfície
ax.plot_surface(X, Y, Z, cmap='viridis')

# Personalização
ax.set_xlabel('Eixo X')
ax.set_ylabel('Eixo Y')
ax.set_zlabel('Eixo Z')
ax.set_title('Gráfico de Superfície 3D')

# Exibir gráfico
plt.show()
