import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Título de la aplicación
st.title('Análisis de Jugadores de Fútbol')

# URL del archivo en GitHub (asegúrate de obtener la URL raw de tu archivo .xlsx en GitHub)
file_url = 'https://raw.githubusercontent.com/usuario/repositorio/master/DataSet_Jugadores_Categorizado-bpa.xlsx'  # Reemplaza esta URL por la correcta

# Cargar el dataset desde la URL
df = pd.read_excel(file_url)

# Mostrar las primeras filas del archivo
st.write("Vista previa de los datos:")
st.write(df.head())

# Limpieza de datos
st.write("Limpiando los datos...")
df.drop_duplicates(inplace=True)
df['goles'].fillna(0, inplace=True)
df['ta'].fillna(0, inplace=True)
media_edad = df[df['edad'] > 0]['edad'].mean()
df['edad'] = df['edad'].replace(0, media_edad)
df['edad'] = df['edad'].astype(int)

# Mostrar la cantidad de valores nulos
st.write("Valores nulos en cada columna:")
st.write(df.isnull().sum())

# Análisis Exploratorio de Datos (EDA)

# Distribución del Valor Mercado
st.write("Distribución del Valor de Mercado:")
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(df['valor'], kde=True, bins=100, ax=ax)
ax.set_title('Distribución del Valor de Mercado de los Jugadores')
st.pyplot(fig)

# Diagrama de caja del Valor de Mercado
st.write("Diagrama de Caja del Valor de Mercado:")
fig, ax = plt.subplots(figsize=(10, 4))
sns.boxplot(x=df['valor'], showmeans=True, ax=ax,
            flierprops={'marker': 'o', 'markerfacecolor': 'red', 'markersize': 7})
ax.set_title('Diagrama de Caja del Valor de Mercado de los Jugadores')
st.pyplot(fig)

# Análisis de variables predictoras (e.g., Goles, Edad, Rating)
st.write("Distribución de Goles y otras variables:")
variables = ['goles', 'asistencias', 'edad', 'altura', 'rating', 'posición auxiliar']
for col in variables:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df[col], kde=True, ax=ax)
    ax.set_title(f'Distribución de {col}')
    st.pyplot(fig)

# Relación entre Variables Predictoras y Valor Mercado
st.write("Relación entre Variables Predictoras y Valor de Mercado:")
for col in variables:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df[col], df['valor'], alpha=0.35, s=30, edgecolors='k', linewidths=0.3)
    m, b = np.polyfit(df[col], df['valor'], deg=1)
    x_fit = np.linspace(df[col].min(), df[col].max(), 100)
    y_fit = m * x_fit + b
    ax.plot(x_fit, y_fit, color='red', linewidth=2)
    ax.set_title(f'Relación entre {col} y Valor de mercado')
    ax.set_xlabel(col)
    ax.set_ylabel('Valor de Mercado (€)')
    st.pyplot(fig)

# Mostrar la matriz de correlación
st.write("Matriz de Correlación entre Variables Numéricas:")
fig, ax = plt.subplots(figsize=(8, 6))
corr = df[['goles', 'asistencias', 'edad', 'altura', 'rating', 'valor', 'posición auxiliar']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

# Comparación de jugadores (Función de recomendación)
st.write("Comparar dos jugadores:")
player1_index = st.number_input('Ingrese el índice del primer jugador', min_value=0, max_value=len(df)-1)
player2_index = st.number_input('Ingrese el índice del segundo jugador', min_value=0, max_value=len(df)-1)

def compare_players(player1_index, player2_index, dataframe):
    player1_features = dataframe.loc[player1_index]
    player2_features = dataframe.loc[player2_index]
    
    # Extraer los nombres de los jugadores
    player1_name = player1_features['jugador']  # Asume que la columna de nombres se llama 'jugador'
    player2_name = player2_features['jugador']  # Asume que la columna de nombres se llama 'jugador'
    
    # Mostrar las características de los jugadores seleccionados
    st.write(f"Características del Jugador 1 ({player1_name} - Índice: {player1_index}):")
    st.write(player1_features)
    st.write(f"Características del Jugador 2 ({player2_name} - Índice: {player2_index}):")
    st.write(player2_features)
    
    score1 = 0
    score2 = 0

    weights = {
        'rating': 0.3, 'edad': 0.1, 'pt': 0.1, 'posición auxiliar': 0.05,
        'altura': 0.05, 'asistencias': 0.15, 'pj': 0.05, 'goles': 0.15, 'ta': -0.05
    }

    # Comparación de cada característica
    comparison_details = []
    for feature, weight in weights.items():
        if weight > 0:
            if player1_features[feature] > player2_features[feature]:
                score1 += weight
                comparison_details.append(f"{feature}: {player1_name} tiene mayor valor ({player1_features[feature]}) que {player2_name} ({player2_features[feature]})")
            elif player2_features[feature] > player1_features[feature]:
                score2 += weight
                comparison_details.append(f"{feature}: {player2_name} tiene mayor valor ({player2_features[feature]}) que {player1_name} ({player1_features[feature]})")
            else:
                comparison_details.append(f"{feature}: Ambos jugadores tienen el mismo valor ({player1_features[feature]})")
        elif weight < 0:
            if player1_features[feature] < player2_features[feature]:
                score1 += abs(weight)
                comparison_details.append(f"{feature}: {player1_name} tiene menor valor ({player1_features[feature]}) que {player2_name} ({player2_features[feature]})")
            elif player2_features[feature] < player1_features[feature]:
                score2 += abs(weight)
                comparison_details.append(f"{feature}: {player2_name} tiene menor valor ({player2_features[feature]}) que {player1_name} ({player1_features[feature]})")
            else:
                comparison_details.append(f"{feature}: Ambos jugadores tienen el mismo valor ({player1_features[feature]})")

    # Mostrar los detalles de la comparación
    st.write("Detalles de la comparación:")
    for detail in comparison_details:
        st.write(detail)

    # Determinar cuál jugador es mejor
    if score1 > score2:
        return f"{player1_name} es mejor que {player2_name}."
    elif score2 > score1:
        return f"{player2_name} es mejor que {player1_name}."
    else:
        return f"Los jugadores {player1_name} y {player2_name} son comparables."

if player1_index is not None and player2_index is not None:
    result = compare_players(player1_index, player2_index, df)
    st.write(result)
