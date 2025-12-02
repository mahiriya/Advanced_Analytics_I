
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

# Load State and City Maps (for target encoding)
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

# Load State-City Average Size and Price_per_SqFt
try:
    with open('state_city_size_avg.json', 'r') as file:
        state_city_size_avg_data = json.load(file)
except FileNotFoundError:
    st.error("State-City average size file 'state_city_size_avg.json' not found. Please ensure it's in the same directory.")
    st.stop()

try:
    with open('state_city_price_per_sqft_avg.json', 'r') as file:
        state_city_price_per_sqft_avg_data = json.load(file)
except FileNotFoundError:
    st.error("State-City average price per sqft file 'state_city_price_per_sqft_avg.json' not found. Please ensure it's in the same directory.")
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
    # Dynamic defaults for Size_in_SqFt and Price_per_SqFt
    # Default to global mean if state/city specific average not found
    # (These default values should ideally come from overall training data stats if possible)
    global_default_size_in_sqft = 2500.41 # Placeholder, replace with actual global mean if available
    global_default_price_per_sqft = 0.10 # Placeholder, replace with actual global mean if available

    # State and City selection by name
    all_states = sorted(list(state_map.keys()))
    selected_state_name = st.sidebar.selectbox('State', all_states)

    # Get cities for selected state
    cities_in_state = sorted(list(state_city_size_avg_data.get(selected_state_name, {}).keys()))
    if not cities_in_state:
        cities_in_state = sorted(list(city_map.keys())) # Fallback to all cities if state not found in avg data

    selected_city_name = st.sidebar.selectbox('City', cities_in_state)

    # Retrieve dynamic averages based on selected State and City
    current_size_avg = state_city_size_avg_data.get(selected_state_name, {}).get(selected_city_name, global_default_size_in_sqft)
    current_price_per_sqft_avg = state_city_price_per_sqft_avg_data.get(selected_state_name, {}).get(selected_city_name, global_default_price_per_sqft)

    size_in_sqft = st.sidebar.number_input('Size in SqFt', value=float(current_size_avg), min_value=100.0, max_value=10000.0)
    price_per_sqft = st.sidebar.number_input('Price per SqFt (in Lakhs)', value=float(current_price_per_sqft_avg), min_value=0.01, max_value=1.0, format="%.2f")

    bhk = st.sidebar.selectbox('BHK', [1, 2, 3, 4, 5])
    nearby_schools = st.sidebar.slider('Nearby Schools', 0, 15, 5)
    nearby_hospitals = st.sidebar.slider('Nearby Hospitals', 0, 10, 3)
    parking_space = st.sidebar.selectbox('Parking Space (Yes=1, No=0)', [0, 1])
    security = st.sidebar.selectbox('Security (Yes=1, No=0)', [0, 1])
    year_built = st.sidebar.slider('Year Built', 1950, 2024, 2005)
    floor_no = st.sidebar.slider('Floor Number', 1, 30, 10)
    total_floors = st.sidebar.slider('Total Floors in Building', 1, 40, 20)
    age_of_property = 2024 - year_built # Calculate age from year built
    availability_status = st.sidebar.selectbox('Availability Status (Ready_to_Move=1, Under_Construction=0)', [0, 1])

    # Get encoded values from the maps for State and City
    state_encoded_val = state_map.get(selected_state_name, 0) # Default to 0 or a sensible value if not found
    city_encoded_val = city_map.get(selected_city_name, 0) # Default to 0 or a sensible value if not found

    # Dummy variables for categorical features (set defaults, then update based on user input for those available)
    # Property Type
    property_type = st.sidebar.selectbox('Property Type', ['Apartment', 'Independent House', 'Villa'])
    prop_type_ind_house = 1 if property_type == 'Independent House' else 0
    prop_type_villa = 1 if property_type == 'Villa' else 0

    # Furnished Status
    furnished_status = st.sidebar.selectbox('Furnished Status', ['Furnished', 'Semi-furnished', 'Unfurnished'])
    furn_status_semi = 1 if furnished_status == 'Semi-furnished' else 0
    furn_status_unfurn = 1 if furnished_status == 'Unfurnished' else 0

    # Public Transport Accessibility
    pt_accessibility = st.sidebar.selectbox('Public Transport Accessibility', ['High', 'Low', 'Medium'])
    pta_low = 1 if pt_accessibility == 'Low' else 0
    pta_medium = 1 if pt_accessibility == 'Medium' else 0

    # Facing
    facing = st.sidebar.selectbox('Facing', ['East', 'North', 'South', 'West'])
    facing_north = 1 if facing == 'North' else 0
    facing_south = 1 if facing == 'South' else 0
    facing_west = 1 if facing == 'West' else 0

    # Owner Type
    owner_type = st.sidebar.selectbox('Owner Type', ['Owner', 'Builder', 'Broker'])
    owner_type_builder = 1 if owner_type == 'Builder' else 0
    owner_type_owner = 1 if owner_type == 'Owner' else 0

    # Amenities (Simplified for selection, assuming most properties have these)
    clubhouse = st.sidebar.checkbox('Clubhouse', value=True)
    garden = st.sidebar.checkbox('Garden', value=True)
    gym = st.sidebar.checkbox('Gym', value=True)
    playground = st.sidebar.checkbox('Playground', value=True)
    pool = st.sidebar.checkbox('Pool', value=True)


    data = {
        'Size_in_SqFt': size_in_sqft,
        'Price_per_SqFt': price_per_sqft,
        'Year_Built': year_built,
        'Floor_No': floor_no,
        'Total_Floors': total_floors,
        'Age_of_Property': age_of_property,
        'Nearby_Schools': nearby_schools,
        'Nearby_Hospitals': nearby_hospitals,
        'Parking_Space': parking_space,
        'Security': security,
        'Availability_Status': availability_status,
        'State_Encoded': state_encoded_val,
        'City_Encoded': city_encoded_val,
        'Property_Type_Independent House': prop_type_ind_house,
        'Property_Type_Villa': prop_type_villa,
        'Furnished_Status_Semi-furnished': furn_status_semi,
        'Furnished_Status_Unfurnished': furn_status_unfurn,
        'Public_Transport_Accessibility_Low': pta_low,
        'Public_Transport_Accessibility_Medium': pta_medium,
        'Facing_North': facing_north,
        'Facing_South': facing_south,
        'Facing_West': facing_west,
        'Owner_Type_Builder': owner_type_builder,
        'Owner_Type_Owner': owner_type_owner,
        'BHK_2': 0,
        'BHK_3': 0,
        'BHK_4': 0,
        'BHK_5': 0,
        'Clubhouse': int(clubhouse),
        'Garden': int(garden),
        'Gym': int(gym),
        'Playground': int(playground),
        'Pool': int(pool),
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
if st.sidebar.button('Predict Price (Regression)', key='reg_pred_btn'):
    regression_prediction = regression_model.predict(input_df)[0]
    st.subheader('💰 Predicted Price (Regression):')
    st.metric(label="Price in Lakhs", value=f"{regression_prediction:.2f} Lakhs")
    st.info("This is the estimated continuous price of the property.")

# Classification Prediction
if st.sidebar.button('Predict Price Category (Classification)', key='clf_pred_btn'):
    classification_prediction_encoded = classification_model.predict(input_df)[0]
    # Map the encoded prediction back to a human-readable label
    predicted_category = class_labels[int(classification_prediction_encoded)] # Ensure index is integer
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
