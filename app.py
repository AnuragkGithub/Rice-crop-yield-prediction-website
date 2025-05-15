import streamlit as st
import pandas as pd
import pickle

# Load model and encoders
@st.cache_resource
def load_model_and_encoders():
    with open("rice_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    return model, encoders

model, encoders = load_model_and_encoders()

st.title("🌾 Rice Crop Production Predictor")
st.markdown("Predict rice yield based on environmental and regional factors.")

st.sidebar.header("Enter Input Features")

# Sidebar inputs
state = st.sidebar.selectbox("State", ['Andhra Pradesh', 'Tamil Nadu', 'West Bengal', 'Karnataka', 'Other'])
district = st.sidebar.text_input("District", "DefaultDistrict")
season = st.sidebar.selectbox("Season", ['Kharif', 'Rabi', 'Whole Year', 'Summer', 'Winter'])

year = st.sidebar.slider("Crop Year", 1997, 2014, 2010)
temperature = st.sidebar.slider("Temperature (°C)", 15.0, 45.0, 30.0)
humidity = st.sidebar.slider("Humidity (%)", 10.0, 100.0, 70.0)
soil_moisture = st.sidebar.slider("Soil Moisture (%)", 0.0, 100.0, 50.0)
area = st.sidebar.number_input("Area (in hectares)", min_value=1.0, value=100.0)

 # Or any other district name that exists in your dataset

# Prepare input data as DataFrame
input_df = pd.DataFrame({
    'State_Name': [state],
    'District_Name': [district],
    'Crop_Year': [year],
    'Season': [season],
    'Temperature': [temperature],
    'Humidity': [humidity],
    'Soil_Moisture': [soil_moisture],
    'Area': [area]
})

# Encode categorical inputs
try:
    input_df['State_Name'] = encoders['state'].transform(input_df['State_Name'])
except ValueError:
    input_df['State_Name'] = encoders['state'].transform([state])[0]

try:
    input_df['District_Name'] = encoders['district'].transform(input_df['District_Name'])
except ValueError:
    input_df['District_Name'] = -1  # Assign a fallback value if unseen label
    st.warning(f"District '{district}' is not recognized. Using default encoding.")
    
try:
    input_df['Season'] = encoders['season'].transform(input_df['Season'])
except ValueError:
    input_df['Season'] = encoders['season'].transform([season])[0]

# Predict production when button is pressed
if st.button("Predict Production"):
    try:
        prediction = model.predict(input_df)[0]
        st.success(f"✅ Estimated Rice Production: **{prediction:.2f} tonnes**")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
