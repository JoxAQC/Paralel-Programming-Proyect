import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import tkinter as tk


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

# Entrenamiento del modelo
X = df[['Anio']]
y = df['Pasajeros']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# Definir función para hacer predicciones y generar gráficos
def predict_and_plot(year, station):
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
target_comparacion = "Atocongo"

def plot_station_prediction(year, station):
    # Predecir para el año deseado
    y_pred = model.predict([[year]])

    # Generar gráfico de predicción
    plt.plot(df['Anio'], df['Pasajeros'], label='Datos reales')
    plt.scatter([year], [y_pred], color='red', label=f'Predicción - {station}')
    plt.title(f'Predicción de Pasajeros para {station} en {year}')
    plt.xlabel('Anio')
    plt.ylabel('Pasajeros')
    plt.legend()
    plt.show()

def plot_station_comparison(station1, station2):
    # Comparar datos de dos estaciones
    df_station1 = df[df['Estacion'] == station1]
    df_station2 = df[df['Estacion'] == station2]

    plt.plot(df_station1['Anio'], df_station1['Pasajeros'], label=f'Datos - {station1}')
    plt.plot(df_station2['Anio'], df_station2['Pasajeros'], label=f'Datos - {station2}')
    plt.title(f'Comparación de Pasajeros entre {station1} y {station2}')
    plt.xlabel('Anio')
    plt.ylabel('Pasajeros')
    plt.legend()
    plt.show()

def analyze_trends():
    # Analizar tendencias en los datos
    df_mean = df.groupby('Anio')['Pasajeros'].mean().reset_index()
    
    plt.plot(df_mean['Anio'], df_mean['Pasajeros'], label='Media de Pasajeros por Año')
    plt.title('Tendencia de Pasajeros a lo largo de los Años')
    plt.xlabel('Anio')
    plt.ylabel('Pasajeros')
    plt.legend()
    plt.show()

def plot_residuals():
    # Visualizar residuos del modelo
    y_pred_train = model.predict(X_train)
    residuals = y_train - y_pred_train

    plt.scatter(X_train['Anio'], residuals, label='Residuos')
    plt.axhline(0, color='red', linestyle='--', label='Línea base')
    plt.title('Residuos del Modelo de Regresión')
    plt.xlabel('Anio')
    plt.ylabel('Residuos')
    plt.legend()
    plt.show()

def plot_feature_importance():
    # Visualizar importancia de las características en el modelo
    feature_importance = model.feature_importances_
    features = X_train.columns

    plt.bar(features, feature_importance)
    plt.title('Importancia de las Características en el Modelo')
    plt.xlabel('Características')
    plt.ylabel('Importancia')
    plt.show()

def run_parallel_functions():
    parallel_functions = [
        (plot_station_prediction, (target_year, target_station)),
        (plot_station_comparison, (target_comparacion, target_station)),
        (analyze_trends, ()),
        (plot_residuals, ()),
        (plot_feature_importance, ())
        # Agrega más funciones según sea necesario
    ]

    results = []

    def update_results():
        for func, args in parallel_functions:
            result = func(*args)
            results.append(result)

    # Ejecutar funciones en paralelo
    with ThreadPoolExecutor() as executor:
        executor.submit(update_results)

    # Actualizar la interfaz gráfica después de un tiempo
    root.after(1000, lambda: display_results(results))

def display_results(results):
    # Crear una ventana para mostrar resultados
    result_window = tk.Toplevel(root)
    result_window.title("Resultados")

    # Mostrar resultados en la ventana
    for result in results:
        tk.Label(result_window, text=result).pack()

# Crear una ventana principal de tkinter
root = tk.Tk()
root.title("Ejecución en Paralelo")

# Botón para ejecutar funciones en paralelo
button = tk.Button(root, text="Ejecutar en Paralelo", command=run_parallel_functions)
button.pack(pady=20)

# Ejecutar la interfaz de usuario
root.mainloop()