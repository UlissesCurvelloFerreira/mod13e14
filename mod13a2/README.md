# Projetos Python - Análise e Visualização de Dados

## Descrição
Este repositório contém scripts Python para análise e manipulação de dados. Os principais recursos incluem a leitura de planilhas, substituição de valores ausentes, geração de gráficos e análise de correlações.

## Scripts

### 1. **exibir.py**
Este script lê e exibe dados de uma planilha Excel (`dados.xlsx`) e realiza as seguintes operações:
- Verifica a existência do arquivo.
- Substitui valores ausentes na coluna "Precipitação (mm)" pela média.
- Substitui valores ausentes na coluna "Umidade Relativa (%)" pela mediana.
- Adiciona uma coluna "Amplitude Térmica", que é a diferença entre a "Temperatura Máxima (°C)" e "Temperatura Mínima (°C)".
- Filtra as cidades com temperatura máxima acima de 30°C.
- Reordena as colunas para a ordem desejada.
- Salva as alterações na planilha original e em uma nova planilha (`new_dados.xlsx`).

### 2. **atividade1.py**
Este script gera um gráfico de linha que mostra a evolução da temperatura ao longo de um dia. Ele usa dados fictícios de temperatura, com a hora do dia no eixo X e a temperatura (em °C) no eixo Y.

### 3. **atividade2.py**
Este script cria três gráficos baseados em dados simulados de vendas de uma loja:
- **Gráfico 1**: Gráfico de barras para o total de vendas por dia da semana.
- **Gráfico 2**: Gráfico de dispersão que relaciona o número de clientes com o total de vendas.
- **Gráfico 3**: Heatmap que mostra a correlação entre as variáveis "Vendas", "Clientes" e "Lucro".

## Requisitos
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`

Para instalar as dependências, utilize o seguinte comando:


