import streamlit as st
import pandas as pd
import pickle

# Load model and columns
model = pickle.load(open('house_price_model.pkl', 'rb'))
model_columns = pickle.load(open('model_columns.pkl', 'rb'))

st.title('🏠 Bangalore House Price Predictor')

# Get user inputs
st.header('Enter House Details:')

sqft = st.number_input('Total Square Feet', min_value=100, value=1000)
bath = st.number_input('Number of Bathrooms', min_value=1, value=2)
bhk = st.number_input('BHK (Bedrooms)', min_value=1, value=2)
balcony = st.number_input('Number of Balconies', min_value=0, value=1)

# Availability
availability = st.selectbox('Availability', ['Ready To Move', 'Under Construction'])

# For location - show top 10 most common + Other option
top_locations = [
    'Whitefield', 'Electronic City', 'Marathahalli', 'BTM Layout',
    'Sarjapur Road', 'Hebbal', 'Yeshwanthpur', 'Rajaji Nagar',
    'Banashankari', 'JP Nagar'
]

location = st.selectbox('Location', top_locations + ['Other Location'])

if st.button('💰 Predict Price'):
    # Prepare input data
    input_data = pd.DataFrame(columns=model_columns)
    input_data.loc[0] = 0  # Initialize with zeros
    
    # Fill basic features (removed price_per_sqft)
    input_data.loc[0, 'total_sqft_cleaned'] = sqft
    input_data.loc[0, 'bath'] = bath
    input_data.loc[0, 'bhk'] = bhk
    input_data.loc[0, 'balcony'] = balcony
    input_data.loc[0, 'availability_encoded'] = 1 if availability == 'Ready To Move' else 0
    
    # Handle location encoding
    if location != 'Other Location':
        location_column = f'location_reduced_{location}'
        if location_column in model_columns:
            input_data.loc[0, location_column] = 1
    
    # Make prediction
    try:
        prediction = model.predict(input_data)[0]
        
        st.success(f'🎯 Predicted Price: ₹{prediction:.2f} Lakhs')
        st.info(f'💰 That\'s approximately ₹{prediction*100000:.0f}')
        
        # Show what was predicted
        st.write('**Prediction based on:**')
        st.write(f'• {sqft} sq ft, {bhk} BHK, {bath} bathrooms')
        st.write(f'• {balcony} balconies, {availability}')
        st.write(f'• Location: {location}')
        
    except Exception as e:
        st.error('Something went wrong with prediction!')
        st.write('Error details:', str(e))