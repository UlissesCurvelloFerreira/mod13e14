import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt

# Dados fictícios com mais critérios
vehicle_data = {
    'potencia': [80, 100, 90, 120, 70, 130, 110, 85, 95, 105, 150, 160, 200, 180, 110],
    'peso': [900, 1200, 1100, 1500, 800, 1600, 1400, 950, 1000, 1100, 1400, 1800, 2200, 2100, 1600],
    'consumo': [15, 12, 14, 10, 16, 9, 11, 14, 13, 12, 8, 7, 6, 9, 10],
    'n_portas': [4, 4, 4, 5, 3, 5, 4, 4, 4, 5, 4, 5, 5, 5, 4],
    'label': [1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1]
}

# Convertendo para DataFrame
vehicle_df = pd.DataFrame(vehicle_data)

# Separando variáveis independentes (X) e dependente (y)
X = vehicle_df[['potencia', 'peso', 'consumo', 'n_portas']]
y = vehicle_df['label']

# Criando e treinando o modelo
model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

# Prevendo a classificação para todos os veículos
predictions = model.predict(X)

# Adicionando as previsões ao DataFrame
vehicle_df['predicao'] = predictions

# Exibindo os veículos classificados como "Econômico" (1) e "Não Econômico" (0)
economicos = vehicle_df[vehicle_df['predicao'] == 1]
nao_economicos = vehicle_df[vehicle_df['predicao'] == 0]

print("Veículos classificados como Econômico:")
print(economicos)

print("\nVeículos classificados como Não Econômico:")
print(nao_economicos)


# Função para customizar os rótulos e as cores da árvore
def custom_tree_plot(tree_model, feature_names, class_names):
    fig, ax = plt.subplots(figsize=(12, 8))
    tree.plot_tree(
        tree_model,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        fontsize=12,
        max_depth=4,
        impurity=False,
        proportion=False
    )
    
    # Adicionando cor personalizada aos nós
    for i in range(len(ax.collections)):
        patch = ax.collections[i]
        if i % 2 == 0:  # Nodes (True)
            patch.set_edgecolor('blue')  # Azul para 'Econômico'
        else:  # Leaves (False)
            patch.set_edgecolor('red')  # Vermelho para 'Não Econômico'
    
    plt.show()

# Visualizando a árvore de decisão com as customizações
custom_tree_plot(model, ['Potência', 'Peso', 'Consumo', 'Número de Portas'], ['Não Econômico', 'Econômico'])
