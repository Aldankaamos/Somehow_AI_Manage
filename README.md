![logo](https://github.com/Aldankaamos/Somehow_AI_Manage/assets/93347758/19c9ba51-633c-4c05-9a27-e8053d4ab5af)

# Somehow_AI_Manage

Proyecto de Título - Marcelo Guzmán  
Repositorio: [https://github.com/Aldankaamos/Somehow_AI_Manage](https://github.com/Aldankaamos/Somehow_AI_Manage)

## Descripción

**Somehow_AI_Manage** es una plataforma de gestión inteligente que utiliza modelos de redes neuronales recurrentes (RNN) para el análisis y predicción de series temporales, orientada a la toma de decisiones en entornos empresariales y de gestión de recursos. El sistema permite cargar datos históricos, entrenar modelos de IA y evaluar su desempeño de manera iterativa y automatizada.

Este proyecto fue desarrollado como parte del trabajo de título para optar al grado de Ingeniero Civil en Computación e Informática.

## Características principales

- **Carga y preprocesamiento de datos**: Lectura de archivos CSV, normalización y preparación de datos para entrenamiento.
- **Entrenamiento de modelos RNN**: Implementación de redes neuronales recurrentes con TensorFlow/Keras, incluyendo capas SimpleRNN, Dropout y Dense.
- **Evaluación iterativa**: Entrenamiento y evaluación en múltiples iteraciones y tamaños de conjunto de entrenamiento, con cálculo de métricas como MSE.
- **Exportación de resultados**: Guardado de métricas y predicciones en archivos CSV para su análisis posterior.
- **Visualización**: (Opcional) Gráficas de resultados usando Matplotlib.

## Estructura del repositorio

- `src/` — Código fuente principal del sistema.
- `data/` — Archivos de datos de ejemplo (CSV).
- `notebooks/` — Jupyter Notebooks para experimentación y visualización.
- `results/` — Resultados de las pruebas y métricas exportadas.
- `README.md` — Este archivo.

## Instalación

1. Clona el repositorio:
   ```sh
   git clone https://github.com/Aldankaamos/Somehow_AI_Manage.git
   cd Somehow_AI_Manage
   ```

2. Instala las dependencias:
   ```sh
   pip install -r requirements.txt
   ```

## Uso

1. Coloca tus archivos de datos históricos en la carpeta `data/`.
2. Ejecuta el script principal para entrenar y evaluar el modelo:
   ```sh
   python src/main.py
   ```
3. Los resultados (MSE de entrenamiento y prueba, predicciones) se guardarán en la carpeta `results/`.

## Ejemplo de flujo de trabajo

1. **Carga de datos**: El sistema lee un archivo CSV, convierte la columna de fechas a formato datetime y normaliza los valores de apertura.
2. **Entrenamiento**: Se divide el conjunto de datos en entrenamiento y prueba, se crean secuencias para la RNN y se entrena el modelo en varias iteraciones.
3. **Evaluación**: Se calcula el error cuadrático medio (MSE) para cada iteración y se exportan los resultados.
4. **Visualización**: (Opcional) Se pueden generar gráficos de las predicciones y métricas.

## Tecnologías utilizadas

- Python 3.x
- TensorFlow / Keras
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

## Estructura del modelo RNN

- Capas SimpleRNN (con return_sequences según corresponda)
- Capas Dropout para evitar overfitting
- Capa Dense de salida
- Optimización con Adam y función de pérdida MSE

## Créditos

Desarrollado por Marcelo Guzmán como proyecto de título.  
Universidad de O'Higgins (UOH).

## Licencia

Este proyecto se distribuye bajo la licencia MIT.
