import os
import numpy as np
import json
from tensorflow.keras.models import model_from_json
import streamlit as st

# Define the directory where the model files are located
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')

# Loading the saved models for each zone
loaded_models = {
    'Quads': {
        'architecture': json.load(open(os.path.join(MODEL_DIR, 'model_architecture_Quads.json'), 'r')),
        'weights': os.path.join(MODEL_DIR, 'model_weights_Quads.weights.h5')
    },
    'Smir': {
        'architecture': json.load(open(os.path.join(MODEL_DIR, 'model_architecture_Smir.json'), 'r')),
        'weights': os.path.join(MODEL_DIR, 'model_weights_Smir.weights.h5')
    },
    'Boussafou': {
        'architecture': json.load(open(os.path.join(MODEL_DIR, 'model_architecture_Boussafou.json'), 'r')),
        'weights': os.path.join(MODEL_DIR, 'model_weights_Boussafou.weights.h5')
    }
}

# Function to load and compile models
def load_and_compile_model(architecture, weights):
    model = model_from_json(json.dumps(architecture))  # Convert dict to JSON string
    model.load_weights(weights)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])  # Adjust optimizer and loss as needed
    return model

# Load and compile models
for zone, model_info in loaded_models.items():
    loaded_models[zone]['model'] = load_and_compile_model(model_info['architecture'], model_info['weights'])

# Print the input shape expected by the model
for zone, model_info in loaded_models.items():
    print(f"Input shape expected by model '{zone}': {loaded_models[zone]['model'].input_shape}")

# Function to make predictions
def predict_energy_consumption(zone, Temp, Hum, Wind, hour, dayofweek, quarter, month, year, dayofyear):
    # Make prediction using the corresponding model for the selected zone
    model = loaded_models[zone]['model']
    
    # Include SMA columns for the selected zone
    input_data = [Temp, Hum, Wind, hour, dayofweek, quarter, month, year, dayofyear]
    input_data += [0] * 3  # Placeholder for SMA columns
    
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    
    # Print the shape of the input data before prediction
    print(f"Shape of input data: {input_data_reshaped.shape}")
    
    prediction = model.predict(input_data_reshaped)
    return prediction[0]

# Streamlit UI
st.title('Energy Consumption Prediction')

# Dropdown menu to select the zone
zone = st.selectbox('Select Zone', ['Quads', 'Smir', 'Boussafou'])

# Input fields for user to enter data
Temp = st.number_input('Temperature', value=25.0, step=0.1)
Hum = st.number_input('Humidity', value=50.0, step=0.1)
Wind = st.number_input('Wind Speed', value=10.0, step=0.1)
hour = st.number_input('Hour', value=12, min_value=0, max_value=23, step=1)
dayofweek = st.number_input('Day of Week', value=3, min_value=0, max_value=6, step=1)
quarter = st.number_input('Quarter', value=1, min_value=1, max_value=4, step=1)
month = st.number_input('Month', value=6, min_value=1, max_value=12, step=1)
year = st.number_input('Year', value=2022, min_value=1900, max_value=2100, step=1)
dayofyear = st.number_input('Day of Year', value=175, min_value=1, max_value=366, step=1)

# Button to trigger prediction
if st.button('Predict'):
    prediction = predict_energy_consumption(zone, Temp, Hum, Wind, hour, dayofweek, quarter, month, year, dayofyear)
    # Convert prediction to a scalar value before formatting
    scalar_prediction = prediction.item()
    st.success(f'Predicted Energy Consumption for {zone}: {scalar_prediction:.2f} KW')
