import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# Cargar el dataset con especificación de codificación y delimitador
file_path = "linea1_pasajeros.csv"
df = pd.read_csv(file_path, encoding='latin-1', delimiter=';')

# Verificar las columnas del DataFrame
print(df.columns)

# Asegurarse de que las columnas están en el DataFrame
required_columns = ['Anio', 'Estacion', 'Pasajeros']
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f'La columna {col} no se encuentra en el DataFrame.')

# Preprocesamiento de datos
df['Anio'] = df['Anio'].astype('float32')
df['Pasajeros'] = df['Pasajeros'].astype('float32')

# Crear modelo de regresión
model = RandomForestRegressor(n_estimators=100, max_depth=10)

# Definir función para hacer predicciones y generar gráficos
def predict_and_plot(year, station):
    # Filtrar datos hasta el año deseado
    df_train = df[df['Anio'] <= year]
    
    # Separar los datos de entrenamiento y prueba
    X_train = df_train[['Anio']]
    y_train = df_train['Pasajeros']

    # Entrenar el modelo
    model.fit(X_train, y_train)

    # Predecir para el año deseado
    y_pred = model.predict([[year]])
    print(y_pred)

    # Generar gráfico
    df_plot = df[df['Estacion'] == station]
    plt.plot(df_plot['Anio'], df_plot['Pasajeros'], label='Datos reales')
    plt.scatter([year], [y_pred], color='red', label='Predicción')
    plt.title(f'Predicción de Pasajeros para {station} en {year}')
    plt.xlabel('Anio')
    plt.ylabel('Pasajeros')
    plt.legend()
    plt.show()

# Años y estación de interés
target_year = 2025
target_station = "San Borja Sur"

predict_and_plot(target_year,target_station)