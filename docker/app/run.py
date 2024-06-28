import streamlit as st
import threading
import requests
import datetime as dt
import os
import mlflow
import subprocess
import pandas as pd
from streamlit_option_menu import option_menu
import base64

mlflow.set_tracking_uri("http://mlflow:5000")

def fetch_grafana_alerts():
    GRAFANA_API_URL = "http://grafana:3000/api/alertmanager/grafana/api/v2/alerts"
    API_KEY = "glsa_IzOJksMRCilcbRthAxDEYzTyUrxc3RV8_6284ab9c"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    try:
        response = requests.get(GRAFANA_API_URL, headers=headers)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error fetching Grafana alerts: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching Grafana alerts: {e}")
        return None

def display_alerts(alerts):
    if alerts:
        st.subheader("Alertas de Grafana")
        
        # Preparar los datos para la tabla
        data = []
        for alert in alerts:
            alertname = alert['labels'].get('alertname', 'No name')
            state = alert['status'].get('state', 'Unknown')
            summary = alert['annotations'].get('summary', 'No summary provided')
            description = alert['annotations'].get('description', 'No description provided')
            data.append([alertname, state, summary, description])
        
        # Crear un DataFrame y mostrarlo como tabla
        df = pd.DataFrame(data, columns=["Nombre de Alerta", "Estado", "Resumen", "Descripción"])
        st.table(df)
    else:
        st.write("No hay alertas.")

def get_all_run_ids():
    client = mlflow.tracking.MlflowClient()
    experiment_id_str = str("0")  # Convert experiment_id to string
    runs = client.search_runs([experiment_id_str])  # Pass as a list of strings
    all_run_ids = []
    for run in runs:
        artifacts = client.list_artifacts(run.info.run_id)
        for artifact in artifacts:
            if artifact.is_dir:
                all_run_ids.append([run.info.run_id, artifact.path])
    return all_run_ids

# Función para correr el servidor Flask en un hilo separado
def run_flask_server():
    import flask_server
    flask_server.run_flask()

# Iniciar el servidor Flask en un hilo separado
flask_thread = threading.Thread(target=run_flask_server, daemon=True)
flask_thread.start()

# Configuración inicial de la página
st.set_page_config(
    initial_sidebar_state="expanded",
    page_title="Somehow AI Manage",
    page_icon="📈",
    layout="wide",  
)


# Función para el menú superior y navegación
def top_menu():       
    page = option_menu(
        menu_title=None,  # No mostrar el título del menú
        options=["Página Principal", "Monitoreo", "Rentrenamiento del modelo"],  # Opciones del menú
        icons=["house", "graph-up", "gear"],  # Iconos para cada opción (opcional)
        menu_icon="cast",  # Icono del menú (opcional)
        default_index=0,  # Índice predeterminado
        orientation="horizontal",
        styles={
            "container": {"background-color": "#262730"},
            "icon": {"color": "white", "font-size": "18px"},
            "nav-link": {"color": "white", "font-size": "18px", "margin": "0px"},
            "nav-link-selected": {"background-color": "red", "color": "white"},
        }
    )
    return page

# Página principal
def main_page(all_run_ids):
    st.title("Solicitud de Predicción")
    st.markdown("---")
    a, b = st.columns([7,1])
    st.subheader("Ingrese los datos requeridos")
    model_choice = st.selectbox("Selecciona el modelo a utilizar (Run ID en MLflow)", all_run_ids)
    a, b, c, d = st.columns([4, 1, 4, 1])
    with a:
        uploaded_file_pred = st.file_uploader("Subir dataset para solicitar una predicción al modelo")
    with b: 
        st.write("")
        st.write("")
        st.write("")
        with st.popover("Ver datos 📚", use_container_width=True):
            if uploaded_file_pred is not None:
                data = pd.read_csv(uploaded_file_pred)
                st.write(data)
            else:
                st.error("No data")
    with c:
        uploaded_file_train = st.file_uploader("Subir dataset con el que se entrenó el modelo para realizar calculos de data drift")
    with d:
        st.write("")
        st.write("")
        st.write("")
        with st.popover("Ver datos 📚", use_container_width=True):
            if uploaded_file_train is not None:
                data = pd.read_csv(uploaded_file_train)
                st.write(data) 
            else:
                st.error("No data")
    a, b = st.columns(2)
    with a:

        start_date_pred = st.date_input("Fecha de inicio (opcional)", key="start_date", value=None)
        end_date_pred = st.date_input("Fecha de fin (opcional)", key="end_date", value=None)
    with b:

        start_date_train = st.date_input("Fecha de inicio (opcional)", key="start_date_train", value=None)
        end_date_train = st.date_input("Fecha de fin (opcional)", key="end_date_train", value=None)
    target_column = st.text_input("Especifica la columna objetivo (target) a predecir del set de datos")
    st.markdown("---")
    if st.button("Solicitar Predicción"):
        with st.spinner('Solicitando predicción...'):
            if uploaded_file_pred and uploaded_file_train and model_choice and target_column:
                # Guardar el archivo subido
                file_path_pred = f"/tmp/{uploaded_file_pred.name}"
                with open(file_path_pred, "wb") as f:
                    f.write(uploaded_file_pred.getbuffer())

                file_path_train = f"/tmp/{uploaded_file_train.name}"
                with open(file_path_train, "wb") as f:
                    f.write(uploaded_file_train.getbuffer())
                
                # Convertir fechas de inicio y fin a cadena o establecer a None
                start_date_str_pred = start_date_pred.strftime('%Y-%m-%d') if start_date_pred else None
                end_date_str_pred = end_date_pred.strftime('%Y-%m-%d') if end_date_pred else None
                start_date_str_train = start_date_train.strftime('%Y-%m-%d') if start_date_train else None
                end_date_str_train = end_date_train.strftime('%Y-%m-%d') if end_date_train else None
    
                # Datos para la solicitud
                data = {
                    "run_id": model_choice,
                    "file_path_pred": file_path_pred,
                    "file_path_train": file_path_train,                    
                    "start_date_pred": start_date_str_pred,
                    "end_date_pred": end_date_str_pred,
                    "start_date_train": start_date_str_train,
                    "end_date_train": end_date_str_train,
                    "target": target_column
                }
                
                # Realizar la solicitud de predicción
                response = requests.post("http://localhost:5001/predict", json=data)
                if response.status_code == 200:
                    response_data = response.json()
                    st.success("Predicción y actualización de métricas exitosas.")
                    
                    # Mostrar el gráfico de predicción y el dataframe de resultados uno al lado del otro
                    st.subheader("Resultados de la Predicción")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        col5, col6 = st.columns([1, 20])
                        with col5:
                            # Define the inverted text using HTML and CSS
                            inverted_text = """
                            <div style="transform: rotate(180deg); writing-mode: vertical-lr; text-align: center; padding: 10px; margin-top: 125px;">
                                Value        
                            </div>
                            """
                            # Display the inverted text in Streamlit
                            st.markdown(inverted_text, unsafe_allow_html=True)
                        with col6:
                            st.subheader("Gráfico de Predicción vs Real")
                            plot_image_path = response_data.get('results')
                            if plot_image_path:
                                results_df = pd.read_json(plot_image_path, orient='records')
                                st.line_chart(results_df, x="Date", y=["Actual Value", "Prediction"], color=["#FFFFFF", "#FF0000"], use_container_width=True)
                    
                    with col2:
                        col3, col4 = st.columns(2)
                        with col3:
                            # Mostrar las métricas de evaluación
                            st.subheader("Métricas de Evaluación")
                            metrics_json = response_data.get('metrics')
                            if metrics_json:
                                metrics_df = pd.read_json(metrics_json, orient='records')
                                st.write(metrics_df)
                        with col4:
                            st.subheader("Data de Predicción vs Real")
                            results_json = response_data.get('results')
                            if results_json:
                                results_df = pd.read_json(results_json, orient='records')
                                st.write(results_df)
                    st.balloons()
                else:
                    st.error("Error en la solicitud de predicción.")
            else:
                st.error("Por favor, completa todos los campos requeridos.")
    st.markdown("---")  
    
    if st.button("Informe de data drift usando Evidently AI"):
        with st.spinner('Generando informe...'):
            if uploaded_file_pred and uploaded_file_train and model_choice and target_column:
                # Guardar el archivo subido
                file_path_pred = f"/tmp/{uploaded_file_pred.name}"
                with open(file_path_pred, "wb") as f:
                    f.write(uploaded_file_pred.getbuffer())

                file_path_train = f"/tmp/{uploaded_file_train.name}"
                with open(file_path_train, "wb") as f:
                    f.write(uploaded_file_train.getbuffer())
                
                # Convertir fechas de inicio y fin a cadena o establecer a None
                start_date_str_pred = start_date_pred.strftime('%Y-%m-%d') if start_date_pred else None
                end_date_str_pred = end_date_pred.strftime('%Y-%m-%d') if end_date_pred else None
                start_date_str_train = start_date_train.strftime('%Y-%m-%d') if start_date_train else None
                end_date_str_train = end_date_train.strftime('%Y-%m-%d') if end_date_train else None
    
                # Datos para la solicitud
                data = {
                    "run_id": model_choice,
                    "file_path_pred": file_path_pred,
                    "file_path_train": file_path_train,                    
                    "start_date_pred": start_date_str_pred,
                    "end_date_pred": end_date_str_pred,
                    "start_date_train": start_date_str_train,
                    "end_date_train": end_date_str_train,
                    "target": target_column
                }
                
                # Realizar la solicitud de predicción
                response = requests.post("http://localhost:5001/predict", json=data)
                if response.status_code == 200:
                    response_data = response.json()
                    st.success("Informe generado de forma exitosa.")
                    
                    # Display Evidently report
                    st.subheader("Evidently Report")
                    evidently_report_path = response_data.get('evidently_report')
                    if evidently_report_path:
                        with open(evidently_report_path, "r") as file:
                            evidently_report_html = file.read()
                        st.components.v1.html(evidently_report_html, height=800, scrolling=True)
                    st.balloons()
                else:
                    st.error("Error en la solicitud.")
            else:
                st.error("Por favor, completa todos los campos requeridos.")

# Página de monitoreo
def monitoring_page():
    st.title("Monitoreo")
    st.markdown("---")
    st.subheader("Tableros de Grafana")

    # Links de los tableros de Grafana
    grafana_urls = [
        "http://127.0.0.1:3000/d-solo/ddneqddds2mtca/model-monitoring?orgId=1&refresh=10s&panelId=5",
        "http://127.0.0.1:3000/d-solo/ddneqddds2mtca/model-monitoring?orgId=1&refresh=10s&panelId=7",
        "http://127.0.0.1:3000/d-solo/ddneqddds2mtca/model-monitoring?orgId=1&refresh=10s&panelId=6",
        "http://127.0.0.1:3000/d-solo/ddneqddds2mtca/model-monitoring?orgId=1&refresh=10s&panelId=1",
        "http://127.0.0.1:3000/d-solo/ddneqddds2mtca/model-monitoring?orgId=1&refresh=10s&panelId=3",
        "http://127.0.0.1:3000/d-solo/ddneqddds2mtca/model-monitoring?orgId=1&refresh=10s&panelId=2"
    ]

    # Organizar en dos filas de tres columnas
    rows = [grafana_urls[i:i+3] for i in range(0, len(grafana_urls), 3)]

    for row in rows:
        cols = st.columns(3)
        for col, url in zip(cols, row):
            with col:
                st.markdown(f'<iframe src="{url}" width="100%" height="200" frameborder="0"></iframe>', unsafe_allow_html=True)
    st.markdown("---")
    st.subheader("Alertas generadas en Grafana")
    alerts = fetch_grafana_alerts()
    print(alerts)
    display_alerts(alerts)

# Página de reentrenamiento del modelo
def retrain_model_page(all_run_ids):
    st.title("Reentrenamiento del modelo")
    model_choice = st.selectbox("Seleccionar el modelo a reentrenar de MLflow", all_run_ids)
    a, b = st.columns([4, 1])
    with a:
        uploaded_file = st.file_uploader("Sube la nueva base de datos para realizar el entrenamiento")
    with b: 
        st.write("")
        st.write("")
        st.write("")
        with st.popover("Ver datos 📚", use_container_width=True):
            if uploaded_file is not None:
                data = pd.read_csv(uploaded_file)
                st.write(data)
            else:
                st.error("No data")
    a, b = st.columns(2)
    with a:
        target_column = st.text_input("Especifica la columna objetivo (target) para entrenar el modelo")
        start_date = st.date_input("Fecha de inicio (opcional)", key="retrain_start_date", value=None)
        end_date = st.date_input("Fecha de fin (opcional)", key="retrain_end_date", value=None)
    with b:
        model_name = st.text_input("Ingresa el nombre del modelo (path)")
        epochs = st.text_input("Ingresa la cantidad de epocas de entrenamiento")
        batch_size = st.text_input("Ingresa el tamaño del batch")

    if st.button("Iniciar reentrenamiento"):
        if uploaded_file and model_choice and target_column:
            with st.spinner('Reentrenamiento en proceso...'):
                st.balloons()
                if uploaded_file and model_choice:
                    # Guardar el archivo subido
                    file_path = f"/tmp/{uploaded_file.name}"
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                        
                    # Convertir fechas de inicio y fin a cadena o establecer a None
                    start_date_str = start_date.strftime('%Y-%m-%d') if start_date else None
                    end_date_str = end_date.strftime('%Y-%m-%d') if end_date else None
            
                    # Datos para la solicitud
                    data = {
                        "run_id": model_choice,
                        "file_path": file_path,
                        "start_date": start_date_str,
                        "end_date": end_date_str,
                        "target": target_column,
                        "epochs": int(epochs),
                        "batch_size": int(batch_size),
                        "model_name": model_name
                    }
                    # Realizar la solicitud de predicción
                    response = requests.post("http://localhost:5001/retrain", json=data)
                    if response.status_code == 200:
                        response_data = response.json()
                        st.write(response_data)
                        st.success("Reentrenamiento exitoso.")
                    else:
                        st.error(f"Error al realizar el reentrenamiento. Código de estado HTTP: {response.status_code}")
        else:
            st.error("Por favor, completa todos los campos requeridos.")

# Función principal para correr la aplicación
def main():
    all_run_ids = get_all_run_ids()
    
    page = top_menu()
    
    if page == "Página Principal":
        main_page(all_run_ids)
    elif page == "Monitoreo":
        monitoring_page()
    elif page == "Rentrenamiento del modelo":
        retrain_model_page(all_run_ids)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "logo.png")

    # Define the HTML string with proper image source and styling
    html_string = f"""
    <div style="display: flex; justify-content: right;">
        <img src="data:image/png;base64,{base64.b64encode(open(file_path, 'rb').read()).decode()}" 
             width="275"
             style="display: block; margin: 0 auto;"
        />
    </div>
    """
    # Display the HTML string using st.markdown
    st.sidebar.markdown(html_string, unsafe_allow_html=True)   
    st.sidebar.markdown("---") 
    
    st.sidebar.title("Panel de Control")
    with st.sidebar.popover("Subir modelo 📤", use_container_width=True):
        st.markdown("🚀")
        model_file = st.file_uploader("Seleccione el 'modelo.keras' a subir a MLflow")
        model_file_name = st.text_input("Ingresa el nombre del nuevo modelo (path)")
        if st.button("Subir", use_container_width=True):
            with st.spinner('Subiendo modelo...'):
                if model_file and model_file_name:
                    try:
                        # Datos para la solicitud
                        files = {
                              "model_file": model_file,
                        }
                        data = {
                            "model_file_name": model_file_name,
                        }
                        # Realizar la solicitud de predicción
                        response = requests.post("http://localhost:5001/upload", files=files, data=data)
                        if response.status_code == 200:
                            st.success("Operación exitosa.")
                        else:
                            st.error(f"Error en la operación: {response.status_code}")
                    except Exception as e:
                        st.error(f"Ha ocurrido un error: {e}")
                else:
                    st.error("Por favor, completa todos los campos requeridos.")
    if st.sidebar.button("Actualizar página 🔁", use_container_width=True):
        st.experimental_rerun()
    # URLs
    grafana_url = "http://127.0.0.1:3000/d/ddneqddds2mtca/model-monitoring?orgId=1&refresh=30s"
    mlflow_url = "http://127.0.0.1:5000/"
    
    # Function to generate HTML for opening URL in a new tab
    def open_url_in_new_tab(url):
        js = f"window.open('{url}');"
        html = f"<script>{js}</script>"
        st.components.v1.html(html)
    
    # Sidebar buttons for navigation
    if st.sidebar.button("Ir a Grafana 📊", use_container_width=True):
        open_url_in_new_tab(grafana_url)
    
    if st.sidebar.button("Ir a MLflow 🌊", use_container_width=True):
        open_url_in_new_tab(mlflow_url)
    st.sidebar.markdown("---") 

    # Mostrar el estado de las alertas en la barra lateral
    st.sidebar.subheader("Estado de Alertas 🚦")
    alerts = fetch_grafana_alerts()
    if alerts:
        st.sidebar.write(f":red[- Hay {len(alerts)} alertas activas.]")
        for alert in alerts:
            alertname = alert['labels'].get('alertname', '')
            if alertname in ['Mean Absolute Error', 'Root Mean Square Error', 'r2'] and alert['status'].get('state') == 'active':
                st.sidebar.warning("Advertencia! al menos una alerta crítica esta activa. Se recomienda reentrenar el modelo.")
                break  # Stop after finding the first active alert of interest
    else:
        st.sidebar.write(":green[- 0 alertas activas.]")
        
    # Botón para refrescar notificaciones en la barra lateral
    if st.sidebar.button("Refrescar Alertas", use_container_width=True):
        alerts = fetch_grafana_alerts()
        if alerts:
            st.sidebar.success("- Notificaciones actualizadas.")
        else:
            st.sidebar.warning("- Sin nuevas notificaciones.")
    st.sidebar.markdown("---") 

if __name__ == "__main__":
    # Llamar al comando mlflow ui en una nueva terminal o en segundo plano
    main()
    
      


