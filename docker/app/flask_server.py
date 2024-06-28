import tensorflow as tf
import pandas as pd
import datetime as dt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from prometheus_client import start_http_server, Gauge
import mlflow
import time
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from flask import Flask, request, jsonify
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import matplotlib.pyplot as plt
import time
from tensorflow.keras.models import load_model

# Initialize Flask app
app = Flask(__name__)

# Define Prometheus metrics
rmse_gauge = Gauge('model_rmse', 'Root Mean Square Error')
mae_gauge = Gauge('model_mae', 'Mean Absolute Error')
r2_gauge = Gauge('model_r2', 'R2 Score')
data_drift_gauge = Gauge('data_drift', 'Data Drift')
data_drift_score_gauge = Gauge('data_drift_score', 'Data Drift Score')
data_drift_percentage_gauge = Gauge('data_drift_porcentage', 'Data Drift Percentage')

# Set initial values for metrics
data_drift_score_gauge.set(0.5)  
r2_gauge.set(1.0)   

def generate_and_save_evidently_report(reference_data, current_data):
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_data, current_data=current_data)
    report_file = "/tmp/evidently_report.html"
    report.save_html(report_file)
    return report_file

def get_mlflow_metrics(run_id):
    client = mlflow.tracking.MlflowClient()
    metrics = client.get_run(run_id[0]).data.metrics
    return metrics

def generate_evidently_metrics(reference_data, current_data, target):
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_data, current_data=current_data)
    evidently_metrics = report.as_dict()
    data_drift = evidently_metrics['metrics'][0]['result']['dataset_drift']
    data_drift_score = evidently_metrics['metrics'][1]['result']['drift_by_columns'][target]['drift_score']
    drifted_features = evidently_metrics['metrics'][0]['result']['number_of_drifted_columns']
    total_features = evidently_metrics['metrics'][0]['result']['number_of_columns']
    data_drift_percentage = (drifted_features / total_features) * 100
    return data_drift_score, data_drift_percentage, data_drift

def load_model_func(run_id):
    model_uri = f"runs:/{run_id[0]}/{run_id[1]}"
    model = mlflow.keras.load_model(model_uri)
    config = model.get_config() # Returns pretty much every information about your model
    input_shape = config["layers"][0]["config"]['batch_shape'][1]
    return model, input_shape


def load_and_preprocess_data(file_path_pred, file_path_train, start_date_pred, end_date_pred, start_date_train, end_date_train, target, input_shape):
    df = pd.read_csv(file_path_pred)
    df = df[['Date', target]]
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    df_aux = pd.read_csv(file_path_train)
    df_aux = df_aux[['Date', target]]
    df_aux['Date'] = pd.to_datetime(df_aux['Date'])
    df_aux.set_index('Date', inplace=True)

    df_drift = pd.read_csv(file_path_pred)
    df_drift['Date'] = pd.to_datetime(df_drift['Date'])
    df_drift.set_index('Date', inplace=True)
    
    df_real = pd.read_csv(file_path_train)
    df_real['Date'] = pd.to_datetime(df_real['Date'])
    df_real.set_index('Date', inplace=True)
    
    # Set start_date to the first date in the dataset if not specified
    if start_date_pred is None:
        start_date_pred = df.index.min()
    else:
        start_date_pred = pd.to_datetime(start_date_pred)

    # Set end_date to the last date in the dataset if not specified
    if end_date_pred is None:
        end_date_pred = df.index.max()
    else:
        end_date_pred = pd.to_datetime(end_date_pred)

    # Set start_date to the first date in the dataset if not specified
    if start_date_train is None:
        start_date_train = df_real.index.min()
    else:
        start_date_train = pd.to_datetime(start_date_train)

    # Set end_date to the last date in the dataset if not specified
    if end_date_train is None:
        end_date_train = df_real.index.max()
    else:
        end_date_train = pd.to_datetime(end_date_train)
    
    train_data = df_aux.loc[start_date_train:end_date_train]
    
    train_data_real = df_real.loc[start_date_train:end_date_train]
    drift_data = df_drift.loc[start_date_pred:end_date_pred]

    data = df.loc[start_date_pred:end_date_pred]
    test_data = data
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset_test = test_data[target].values 
    dataset_test = np.reshape(dataset_test, (-1, 1))
    scaled_test = scaler.fit_transform(dataset_test)

    test = []
    for i in range(input_shape, len(scaled_test)):
        test.append(scaled_test[i-input_shape:i, 0])

    test = np.array(test)
    test = np.reshape(test, (test.shape[0], test.shape[1], 1))

    y_test = scaled_test[input_shape:]
    y_test = scaler.inverse_transform(y_test)

    return train_data, test_data, test, y_test, scaler, drift_data, train_data_real

def load_and_preprocess_retrain(file_path, start_date, end_date, target, input_shape):
    df = pd.read_csv(file_path)
    df = df[['Date', target]]
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # Set start_date to the first date in the dataset if not specified
    if start_date is None:
        start_date = df.index.min()
    else:
        start_date = pd.to_datetime(start_date)

    # Set end_date to the last date in the dataset if not specified
    if end_date is None:
        end_date = df.index.max()
    else:
        end_date = pd.to_datetime(end_date)

    data = df.loc[start_date:end_date]
    test_data = data
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset_test = test_data[target].values 
    dataset_test = np.reshape(dataset_test, (-1, 1))
    scaled_test = scaler.fit_transform(dataset_test)

    test = []
    for i in range(input_shape, len(scaled_test)):
        test.append(scaled_test[i-input_shape:i, 0])

    test = np.array(test)
    test = np.reshape(test, (test.shape[0], test.shape[1], 1))

    y_test = scaled_test[input_shape:]
    y_test = scaler.inverse_transform(y_test)

    return test, y_test 


def make_predictions(model, test_data, scaler):
    predictions = model.predict(test_data)
    predictions = scaler.inverse_transform(predictions)
    return predictions

def log_metrics_to_mlflow(rmse, mae, r2, data_drift_score, data_drift_percentage, data_drift, run_id):
    with mlflow.start_run(run_id=run_id[0]):
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("data_drift_score", data_drift_score)
        mlflow.log_metric("data_drift", data_drift)
        mlflow.log_metric("data_drift_percentage", data_drift_percentage)

def update_metrics(run_id, reference_data, current_data, target):
    metrics = get_mlflow_metrics(run_id)
    rmse_gauge.set(metrics.get('rmse', 0))
    mae_gauge.set(metrics.get('mae', 0))
    r2_gauge.set(metrics.get('r2', 0))

    data_drift_score, data_drift_percentage, data_drift  = generate_evidently_metrics(reference_data, current_data, target)
    data_drift_score_gauge.set(data_drift_score)
    data_drift_percentage_gauge.set(data_drift_percentage)
    data_drift_gauge.set(data_drift)
    return data_drift_score, data_drift_percentage, data_drift

def calculate_and_update_metrics(model, test_data, y_true, scaler, run_id, reference_data, current_data, target, drift_data, train_data_real):
    initialTime = time.time()
    y_pred = make_predictions(model, test_data, scaler)
    finalTime = time.time()

    timediff = finalTime - initialTime
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    rmse_gauge.set(rmse)

    mae = mean_absolute_error(y_true, y_pred)
    mae_gauge.set(mae)

    r2 = r2_score(y_true, y_pred)
    r2_gauge.set(r2)

    accuracy = np.mean(np.abs(y_true - y_pred) / y_true < 0.1) * 100  # Assuming 10% range for accuracy
    l1_median = np.median(np.abs(y_true - y_pred))

    data_drift_score, data_drift_percentage, data_drift = update_metrics(run_id, train_data_real, drift_data, target)

    log_metrics_to_mlflow(rmse, mae, r2, data_drift_score, data_drift_percentage, data_drift, run_id)
    
    return y_pred, accuracy, l1_median, rmse, mae, r2, timediff


@app.route('/predict', methods=['POST'])
def trigger_prediction():
    run_id = request.json.get('run_id')
    file_path_pred = request.json.get('file_path_pred')
    start_date_pred = request.json.get('start_date_pred')
    end_date_pred = request.json.get('end_date_pred')
    file_path_train = request.json.get('file_path_train')
    start_date_train = request.json.get('start_date_train')
    end_date_train = request.json.get('end_date_train')
    target = request.json.get('target')  
    
    if not all([run_id, file_path_pred, file_path_train,  target]):
        return jsonify({"error": "run_id, file_path_pred, file_path_train and target are required"}), 400

    model, input_shape = load_model_func(run_id)
    train_data, test_data, test, y_test, scaler, drift_data, train_data_real = load_and_preprocess_data(file_path_pred, file_path_train, start_date_pred, end_date_pred, start_date_train, end_date_train, target, input_shape)
    
    y_pred, accuracy, l1_median, rmse, mae, r2, timediff = calculate_and_update_metrics(model, test, y_test, scaler, run_id, train_data, test_data, target, drift_data, train_data_real)
    
    global current_train_data, current_test_data, current_test, current_y_test, current_model, current_scaler
    current_train_data = train_data
    current_test_data = test_data
    current_test = test
    current_y_test = y_test
    current_model = model
    current_scaler = scaler

    # Convert results dataframe to JSON
    results_df = pd.DataFrame({
        "Date": test_data[input_shape:].index.strftime('%Y-%m-%d %H:%M:%S'),
        "Actual Value": y_test.flatten(),
        "Prediction": y_pred.flatten(),
    })
    results_json = results_df.to_json(orient='records')
    
    # Create metrics dataframe
    metrics_df = pd.DataFrame({
        "Metricas de evaluación obtenidas": ["RMSE", "MAE", "R2", "Accuracy (%)", "L1 Median", "Prediction time(s)"],
        "Value": [rmse, mae, r2, accuracy, l1_median, timediff]
    })
    metrics_json = metrics_df.to_json(orient='records')

    # Generate and save Evidently report
    evidently_report_file = generate_and_save_evidently_report(train_data_real, drift_data)
    
    # Return response with plot, results, and Evidently report
    response_data = {
        "message": "Prediction and metrics update successful",
        #"plot_image": plot_file,
        "results": results_json,
        "metrics": metrics_json,
        "evidently_report": evidently_report_file
    }
    
    return jsonify(response_data)

def retrain_model(run_id, file_path, start_date, end_date, target, epochs, batch_size, model_name):
    # Load model
    model, input_shape = load_model_func(run_id)
    # Load and preprocess data
    test, y_test = load_and_preprocess_retrain(file_path, start_date, end_date, target, input_shape)

    #for layer in model.layers:
        #if hasattr(layer, 'kernel_initializer') and hasattr(layer, 'bias_initializer'):
            #layer.kernel.assign(layer.kernel_initializer(layer.kernel.shape))
            #layer.bias.assign(layer.bias_initializer(layer.bias.shape))
        
    model.compile(
        loss="mse",
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        metrics=["mae"],
    )
    model.fit(test, y_test, epochs=epochs, batch_size=batch_size)  # Example retraining process

    # Start a new MLflow run to log the retrained model
    with mlflow.start_run() as run:
        # Log retrained model to a new path in MLflow
        mlflow.keras.log_model(model, model_name)
        # Get the new run ID generated by MLflow
        new_run_id = run.info.run_id

    # Return response or update any necessary state
    return {"message": "Model retrained successfully", "run_id": new_run_id}

@app.route('/retrain', methods=['POST'])
def trigger_retrain():
    # Extract parameters from POST request
    run_id = request.json.get('run_id')
    file_path = request.json.get('file_path')
    start_date = request.json.get('start_date')
    end_date = request.json.get('end_date')
    target = request.json.get('target')
    epochs = request.json.get('epochs')
    batch_size = request.json.get('batch_size')
    model_name = request.json.get('model_name')

    # Perform retraining
    if not all([run_id, file_path, target]):
        return jsonify({"error": "run_id, file_path, and target are required"}), 400
    
    retrain_result = retrain_model(run_id, file_path, start_date, end_date, target, epochs, batch_size, model_name)
    
    return jsonify(retrain_result)

@app.route('/upload', methods=['POST'])
def trigger_upload():
    # Extract files and form data from POST request
    model_file = request.files['model_file']
    model_file_name = request.form['model_file_name']

    if model_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save the uploaded file to a temporary location
    temp_path = f"/tmp/{model_file.filename}"
    model_file.save(temp_path)

    # Load the Keras model from the file
    model = load_model(temp_path)

    # Start a new MLflow run to log the retrained model
    with mlflow.start_run() as run:
        # Log retrained model to a new path in MLflow
        mlflow.keras.log_model(model, model_file_name)
        # Get the new run ID generated by MLflow
        run_id = run.info.run_id

    return jsonify({"message": "Model uploaded successfully", "run_id": run_id})
    
def run_flask(): 
    mlflow.set_tracking_uri("http://mlflow:5000")
    start_http_server(8000)
    app.run(port=5001)

if __name__ == '__main__':
    run_flask()

