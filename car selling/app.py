import streamlit as st
import joblib
import pandas as pd

# Load the trained model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('car_price_model.pkl')
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

model = load_model()

# App title
st.title("🚗 Car Price Predictor")

# Create form
with st.form("prediction_form"):
    st.header("Basic Information")
    
    col1, col2 = st.columns(2)
    with col1:
        make = st.selectbox("Make", ["Maruti Suzuki", "Hyundai", "Honda", "Toyota", "Tata", "Mahindra", "Kia", "Volkswagen"])
    with col2:
        fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "Hybrid", "LPG", "Electric"])
    
    col1, col2 = st.columns(2)
    with col1:
        kilometers = st.number_input("Kilometers Driven", min_value=0, max_value=500000, step=1000)
    with col2:
        owners = st.number_input("Number of Owners", min_value=0, max_value=5, step=1)
    
    st.header("Car Specifications")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        length = st.number_input("Length (mm)", min_value=0, max_value=6000, step=10)
    with col2:
        width = st.number_input("Width (mm)", min_value=0, max_value=3000, step=10)
    with col3:
        height = st.number_input("Height (mm)", min_value=0, max_value=3000, step=10)
    
    col1, col2 = st.columns(2)
    with col1:
        seating = st.number_input("Seating Capacity", min_value=2, max_value=10, step=1)
    with col2:
        fuel_tank = st.number_input("Fuel Tank Capacity (liters)", min_value=0, max_value=100, step=1)
    
    st.header("Additional Information")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        car_age = st.number_input("Car Age (years)", min_value=0, max_value=30, step=1)
    with col2:
        transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
    with col3:
        seller_type = st.selectbox("Seller Type", ["Individual", "Dealer", "Trustmark Dealer"])
    
    submit_button = st.form_submit_button("Predict Price")

# When form is submitted
if submit_button and model is not None:
    try:
        # Create a DataFrame with the correct feature structure
        input_data = {
            'Kilometers_Driven': kilometers,
            'Owner_Type': owners,
            'Length': length,
            'Width': width,
            'Height': height,
            'Seating_Capacity': seating,
            'Fuel_Tank_Capacity': fuel_tank,
            'Car_Age': car_age,
            'Make_' + make: 1,
            'Fuel_Type_' + fuel_type: 1,
            'Transmission_' + transmission: 1,
            'Seller_Type_' + seller_type: 1
        }
        
        # Convert to DataFrame
        input_df = pd.DataFrame([input_data]).fillna(0)
        
        # Make sure all expected columns are present
        expected_columns = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else None
        if expected_columns is not None:
            # Add missing columns with 0 values
            for col in expected_columns:
                if col not in input_df.columns:
                    input_df[col] = 0
            # Reorder columns to match training data
            input_df = input_df[expected_columns]
        
        # Make prediction
        prediction = model.predict(input_df)
        predicted_price = round(prediction[0], 2)
        
        # Display result
        st.success(f"### Predicted Price: ₹{predicted_price:,.2f}")
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please ensure your model was trained with the correct features.")
elif submit_button and model is None:
    st.error("Model failed to load. Please check the model file.")