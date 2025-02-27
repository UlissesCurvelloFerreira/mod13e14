import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Carregar os dados do arquivo 'automovel.xlsx'
df = pd.read_excel('automovel2.xlsx')

# Separando as variáveis independentes (X) e a variável dependente (y)
X = df[['Categoria', 'Idade', 'Quilometragem']]
y = df['Preco']

# Dividir os dados em conjunto de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definir o pré-processamento para dados numéricos e categóricos
numeric_features = ['Idade', 'Quilometragem']
categorical_features = ['Categoria']

# Transformações: StandardScaler para numéricos e OneHotEncoder para categóricos
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Criar o pipeline com o pré-processamento e o modelo de regressão linear
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])

# Treinar o modelo
pipeline.fit(X_train, y_train)

# Fazer previsões
y_pred = pipeline.predict(X_test)

# Calcular o erro quadrático médio (MSE)
mse = mean_squared_error(y_test, y_pred)

# Exibir resultados de forma mais interessante
print("\n--- Resultados do Modelo ---")
print(f"\nErro Quadrático Médio (MSE): {mse:.2f}\n")

# Exibir os dados reais e as previsões
test_results = X_test.copy()
test_results['Preço Real'] = y_test
test_results['Preço Previsto'] = y_pred

print("\nTabela de Resultados (dados reais e previsões):")
print(test_results)

# Exibir todos os dados com as previsões (treinamento + teste)
all_data = X.copy()
all_data['Preço Real'] = y
all_data['Preço Previsto'] = pipeline.predict(X)
print("\nTabela Completa com Previsões para Todos os Dados:")
print(all_data)

# Obter os coeficientes da regressão e a equação
model = pipeline.named_steps['model']
intercept = model.intercept_
coefficients = model.coef_

# Mostrar a equação de regressão
categories = pipeline.named_steps['preprocessor'].transformers_[1][1].categories_[0]  # Categorias de 'Categoria'

# Construindo a equação
equation = f"Preço = {intercept:.2f} + "
equation += f"{coefficients[0]:.2f} * Idade + {coefficients[1]:.2f} * Quilometragem + "
for i, category in enumerate(categories):
    equation += f"{coefficients[i+2]:.2f} * {category} + "  # Considerando o OneHotEncoder

equation = equation.strip(' +')

print("\nEquação de Regressão Linear:")
print(equation)

# Comparando três valores
sample_data = pd.DataFrame({
    'Categoria': ['Gasolina', 'Diesel', 'Etanol'],
    'Idade': [4, 5, 6],
    'Quilometragem': [60000, 70000, 80000]
})

# Fazer previsões para os valores amostrais
sample_predictions = pipeline.predict(sample_data)

# Exibir as comparações
print("\nComparação de Preço para 3 valores de entrada:")

for i, sample in sample_data.iterrows():
    print(f"\nCategoria: {sample['Categoria']}, Idade: {sample['Idade']} anos, Quilometragem: {sample['Quilometragem']} km -> Preço Previsto: R${sample_predictions[i]:.2f}")
