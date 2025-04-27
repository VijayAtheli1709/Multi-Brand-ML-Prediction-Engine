import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="Used Car Price Predictor",
    page_icon="🚗",
    layout="wide"
)

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Function to load model
def load_model(brand):
    """Load a trained model for a specific car brand"""
    model_path = f'models/{brand}_pipeline.pkl'
    
    if not os.path.exists(model_path):
        return None
        
    try:
        pipeline = joblib.load(model_path)
        return pipeline
    except Exception as e:
        st.error(f"Error loading model for {brand}: {str(e)}")
        return None

# Define brand-to-model mapping
brand_model_map = {
    'Audi': ['A1', 'A6', 'A4', 'A3', 'Q3', 'Q5', 'A5', 'S4', 'Q2', 'A7', 'TT', 'Q7',
             'RS6', 'RS3', 'A8', 'Q8', 'RS4', 'RS5', 'R8', 'SQ5', 'S8', 'SQ7', 'S3',
             'S5', 'A2', 'RS7'],
    'BMW': ['5 Series', '6 Series', '1 Series', '7 Series', '2 Series', '4 Series',
            'X3', '3 Series', 'X5', 'X4', 'i3', 'X1', 'M4', 'X2', 'X6', '8 Series',
            'Z4', 'X7', 'M5', 'i8', 'M2', 'M3', 'M6', 'Z3'],
    'Ford': ['Fiesta', 'Focus', 'Puma', 'Kuga', 'EcoSport', 'C-MAX', 'Mondeo', 'Ka+',
             'Tourneo Custom', 'S-MAX', 'B-MAX', 'Edge', 'Tourneo Connect',
             'Grand C-MAX', 'KA', 'Galaxy', 'Mustang', 'Grand Tourneo Connect',
             'Fusion', 'Ranger', 'Streetka', 'Escort', 'Transit Tourneo'],
    'Hyundai': ['I20', 'Tucson', 'I10', 'IX35', 'I30', 'I40', 'Kona', 'Veloster', 'I800',
                'IX20', 'Ioniq', 'Santa Fe', 'Accent', 'Terracan', 'Getz', 'Amica'],
    'Mercedes': ['SLK', 'S Class', 'SL CLASS', 'G Class', 'GLE Class', 'GLA Class',
                 'A Class', 'B Class', 'GLC Class', 'C Class', 'E Class', 'GL Class',
                 'CLS Class', 'CLC Class', 'CLA Class', 'V Class', 'M Class', 'CL Class',
                 'GLS Class', 'GLB Class', 'X-CLASS', '180', 'CLK', 'R Class', '230', '220',
                 '200'],
    'Skoda': ['Octavia', 'Citigo', 'Yeti Outdoor', 'Superb', 'Kodiaq', 'Rapid',
              'Karoq', 'Fabia', 'Yeti', 'Scala', 'Roomster', 'Kamiq'], 
    'Toyota': ['GT86', 'Corolla', 'RAV4', 'Yaris', 'Auris', 'Aygo', 'C-HR', 'Prius',
               'Avensis', 'Verso', 'Hilux', 'PROACE VERSO', 'Land Cruiser', 'Supra',
               'Camry', 'Verso-S', 'IQ', 'Urban Cruiser'], 
    'Vauxhall': ['Corsa', 'Astra', 'Viva', 'Mokka', 'Mokka X', 'Crossland X', 'Zafira',
                 'Meriva', 'Zafira Tourer', 'Adam', 'Grandland X', 'Antara', 'Insignia',
                 'Ampera', 'GTC', 'Combo Life', 'Vivaro', 'Cascada', 'Kadjar', 'Agila',
                 'Tigra', 'Vectra'],
    'VW': ['T-Roc', 'Golf', 'Passat', 'T-Cross', 'Polo', 'Tiguan', 'Sharan', 'Up',
           'Scirocco', 'Beetle', 'Caddy Maxi Life', 'Caravelle', 'Touareg',
           'Arteon', 'Touran', 'Golf SV', 'Amarok', 'Tiguan Allspace', 'Shuttle',
           'Jetta', 'CC', 'California', 'Caddy Life', 'Caddy', 'Caddy Maxi', 'Eos',
           'Fox']
}

# Check available trained models
available_brands = []
for brand in brand_model_map.keys():
    if os.path.exists(f"models/{brand}_pipeline.pkl"):
        available_brands.append(brand)

# App title and description
st.title("Used Car Price Predictor 🚗")
st.markdown("""
This app predicts the price of used cars based on various features.
Select a car brand and fill in the details to get a price estimate.
""")

if not available_brands:
    st.warning("No trained models found. Please make sure you have trained models in the 'models' directory.")
else:
    # Sidebar for inputs
    st.sidebar.header("Car Details")
    
    # Brand selection (only show brands with available models)
    brand = st.sidebar.selectbox("Select Brand", available_brands)
    
    # Show corresponding models
    available_models = brand_model_map.get(brand, [])
    model = st.sidebar.selectbox("Select Model", available_models)
    
    # Car details
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        year = st.number_input("Year", min_value=1990, max_value=2025, value=2018)
        mileage = st.number_input("Mileage", min_value=0, value=25000, step=1000)
        tax = st.number_input("Annual Tax ($)", min_value=0, value=150)
    
    with col2:
        transmission = st.selectbox("Transmission", ['Manual', 'Automatic', 'Semi-Auto'])
        fuel_type = st.selectbox("Fuel Type", ['Petrol', 'Diesel', 'Hybrid', 'Electric'])
        engine_size = st.slider("Engine Size (L)", 0.5, 5.0, 1.5, step=0.1)
    
    mpg = st.sidebar.slider("Fuel Efficiency (MPG)", 10.0, 100.0, 45.0, step=0.5)
    
    # Predict button
    predict_button = st.sidebar.button("Predict Price", type="primary", use_container_width=True)
    
    # Main content
    if predict_button:
        try:
            # Load the model
            with st.spinner(f"Loading {brand} model..."):
                pipeline = load_model(brand)
            
            if pipeline is None:
                st.error(f"Could not load model for {brand}. Please make sure the model file exists in the 'models' directory.")
            else:
                # Create input dataframe
                input_data = pd.DataFrame({
                    'model': [model],
                    'year': [year],
                    'transmission': [transmission],
                    'mileage': [mileage],
                    'fuelType': [fuel_type],
                    'tax': [tax],
                    'mpg': [mpg],
                    'engineSize': [engine_size]
                })
                
                # Make prediction
                with st.spinner("Calculating price..."):
                    predicted_price = pipeline.predict(input_data)[0]
                
                # Display prediction
                st.success(f"## Estimated Price: ${predicted_price:,.2f}")
                
                # Display car details in a nice format
                st.subheader("Car Details")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Brand", brand)
                    st.metric("Model", model)
                    st.metric("Year", year)
                
                with col2:
                    st.metric("Mileage", f"{mileage:,}")
                    st.metric("Transmission", transmission)
                    st.metric("Engine Size", f"{engine_size}L")
                
                with col3:
                    st.metric("Fuel Type", fuel_type)
                    st.metric("MPG", mpg)
                    st.metric("Annual Tax", f"${tax}")
        
        except Exception as e:
            st.error(f"Error predicting price: {str(e)}")
            st.info("Please make sure you've trained the model for this brand.")

# Footer
st.markdown("""---""")
st.markdown("""
<div style="text-align: center">
    <p>Created with ❤️ | Used Car Price Predictor | © 2025</p>
</div>
""", unsafe_allow_html=True)