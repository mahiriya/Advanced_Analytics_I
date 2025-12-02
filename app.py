
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json

# --- Load Models and Features ---
# Load Regression Model
try:
    with open('lgbm_regression_model.pkl', 'rb') as file:
        regression_model = pickle.load(file)
except FileNotFoundError:
    st.error("Regression model file 'lgbm_regression_model.pkl' not found. Please ensure it's in the same directory.")
    st.stop()

# Load Classification Model
try:
    with open('lgbm_classification_model.pkl', 'rb') as file:
        classification_model = pickle.load(file)
except FileNotFoundError:
    st.error("Classification model file 'lgbm_classification_model.pkl' not found. Please ensure it's in the same directory.")
    st.stop()

# Load Feature Columns
try:
    with open('features_columns.json', 'r') as file:
        feature_columns = json.load(file)
except FileNotFoundError:
    st.error("Feature columns file 'features_columns.json' not found. Please ensure it's in the same directory.")
    st.stop()

# Load State and City Maps
try:
    with open('state_map.json', 'r') as file:
        state_map = json.load(file)
except FileNotFoundError:
    st.error("State map file 'state_map.json' not found. Please ensure it's in the same directory.")
    st.stop()
try:
    with open('city_map.json', 'r') as file:
        city_map = json.load(file)
except FileNotFoundError:
    st.error("City map file 'city_map.json' not found. Please ensure it's in the same directory.")
    st.stop()

# Assuming `le` (LabelEncoder) was used to encode target for classification
# For this example, we'll assume the encoded labels are 0, 1, 2 and you know their mapping.
class_labels = ['Low', 'Medium', 'High'] # Map the encoded labels back to original

# --- Streamlit App Layout ---
st.set_page_config(page_title="Housing Price Predictor", layout="wide")
st.title("🏡 Housing Price Prediction App")
st.markdown("This app predicts housing prices (regression) and classifies price categories (low/medium/high).")

st.sidebar.header('User Input Features')

# --- Function to get user input ---
def get_user_input():
    # Set default values for Size_in_SqFt and Price_per_SqFt (mean from training data)
    default_size_in_sqft = 2500.41 # Mean from X_train['Size_in_SqFt'].mean()
    default_price_per_sqft = 0.10 # Mean from X_train['Price_per_SqFt'].mean()

    bhk = st.sidebar.selectbox('BHK', [1, 2, 3, 4, 5])
    nearby_schools = st.sidebar.slider('Nearby Schools', 0, 15, 5)
    nearby_hospitals = st.sidebar.slider('Nearby Hospitals', 0, 10, 3)
    parking_space = st.sidebar.selectbox('Parking Space (Yes=1, No=0)', [0, 1])
    security = st.sidebar.selectbox('Security (Yes=1, No=0)', [0, 1])

    # State and City selection by name
    selected_state_name = st.sidebar.selectbox('State', sorted(list(state_map.keys())))
    selected_city_name = st.sidebar.selectbox('City', sorted(list(city_map.keys())))

    # Get encoded values from the maps
    state_encoded_val = state_map.get(selected_state_name, 0) # Default to 0 or a sensible value if not found
    city_encoded_val = city_map.get(selected_city_name, 0) # Default to 0 or a sensible value if not found

    data = {
        'Size_in_SqFt': default_size_in_sqft,
        'Price_per_SqFt': default_price_per_sqft,
        'Year_Built': 2005, # Example fixed value, you'd want an input
        'Floor_No': 10, # Example fixed value, you'd want an input
        'Total_Floors': 20, # Example fixed value
        'Age_of_Property': 18, # Example fixed value
        'Nearby_Schools': nearby_schools,
        'Nearby_Hospitals': nearby_hospitals,
        'Parking_Space': parking_space,
        'Security': security,
        'Availability_Status': 1, # Example fixed value (Ready_to_Move)
        'State_Encoded': state_encoded_val,
        'City_Encoded': city_encoded_val,
        'Property_Type_Independent House': 0, # Example dummy variables
        'Property_Type_Villa': 0,
        'Furnished_Status_Semi-furnished': 0,
        'Furnished_Status_Unfurnished': 0,
        'Public_Transport_Accessibility_Low': 0,
        'Public_Transport_Accessibility_Medium': 0,
        'Facing_North': 0,
        'Facing_South': 0,
        'Facing_West': 0,
        'Owner_Type_Builder': 0,
        'Owner_Type_Owner': 0,
        'BHK_2': 0,
        'BHK_3': 0,
        'BHK_4': 0,
        'BHK_5': 0,
        'Clubhouse': 0,
        'Garden': 0,
        'Gym': 0,
        'Playground': 0,
        'Pool': 0,
    }

    # Update dummy variables based on BHK input
    if bhk == 2: data['BHK_2'] = 1
    elif bhk == 3: data['BHK_3'] = 1
    elif bhk == 4: data['BHK_4'] = 1
    elif bhk == 5: data['BHK_5'] = 1

    # Create a DataFrame from the inputs
    features = pd.DataFrame(data, index=[0])

    # Ensure the order of columns matches the training data
    # It's crucial that `feature_columns` from JSON is used here
    features = features[feature_columns]

    return features

# Get user input
input_df = get_user_input()

st.subheader('User Input:')
st.write(input_df)

# --- Make Predictions ---

# Regression Prediction
if st.sidebar.button('Predict Price (Regression)'):
    regression_prediction = regression_model.predict(input_df)[0]
    st.subheader('💰 Predicted Price (Regression):')
    st.metric(label="Price in Lakhs", value=f"{regression_prediction:.2f} Lakhs")
    st.info("This is the estimated continuous price of the property.")

# Classification Prediction
if st.sidebar.button('Predict Price Category (Classification)'):
    classification_prediction_encoded = classification_model.predict(input_df)[0]
    # Map the encoded prediction back to a human-readable label
    predicted_category = class_labels[classification_prediction_encoded]
    st.subheader('📊 Predicted Price Category (Classification):')
    st.success(f"The property is predicted to be in the **{predicted_category}** price category.")
    st.info("This classifies the property into 'Low', 'Medium', or 'High' price ranges.")

st.markdown("""
---
#### How to use this app in Colab:
1. Run the code cell above to create `app.py`.
2. In a new Colab cell, run: `!streamlit run app.py`
3. Click the external URL provided by Streamlit to open the app in a new tab.
""")
