import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Set page title
st.title('Income Prediction Model')
st.write('Enter the following information to predict income category')

# Create input fields for all features
age = st.number_input('Age', min_value=17, max_value=90, value=30)
workclass = st.selectbox('Workclass', ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay'])
fnlwgt = st.number_input('Final Weight', min_value=1, max_value=1500000, value=200000)
education = st.selectbox('Education', ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool'])
education_num = st.number_input('Education Number (1-16)', min_value=1, max_value=16, value=10)
marital_status = st.selectbox('Marital Status', ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'])
occupation = st.selectbox('Occupation', ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces'])
relationship = st.selectbox('Relationship', ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'])
race = st.selectbox('Race', ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'])
sex = st.selectbox('Sex', ['Female', 'Male'])
capital_gain = st.number_input('Capital Gain', min_value=0, max_value=100000, value=0)
capital_loss = st.number_input('Capital Loss', min_value=0, max_value=10000, value=0)
hours_per_week = st.number_input('Hours per Week', min_value=1, max_value=100, value=40)
native_country = st.selectbox('Native Country', ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands'])

# Create predict button
predict_button = st.button('Predict Income')

@st.cache_resource
def load_model():
    with open('log_reg_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

@st.cache_resource
def load_encoders():
    with open('label_encoders.pkl', 'rb') as file:
        encoders = pickle.load(file)
    return encoders

if predict_button:
    try:
        # Load the model and encoders
        model = load_model()
        encoders = load_encoders()
        
        # Create a dictionary of input values
        input_data = {
            'age': age,
            'workclass': workclass,
            'fnlwgt': fnlwgt,
            'education': education,
            'education.num': education_num,
            'marital.status': marital_status,
            'occupation': occupation,
            'relationship': relationship,
            'race': race,
            'sex': sex,
            'capital.gain': capital_gain,
            'capital.loss': capital_loss,
            'hours.per.week': hours_per_week,
            'native.country': native_country
        }
        
        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Encode categorical variables
        categorical_cols = ['workclass', 'education', 'marital.status', 'occupation', 
                          'relationship', 'race', 'sex', 'native.country']
        
        for col in categorical_cols:
            input_df[col] = encoders[col].transform(input_df[col])
        
        # Make prediction
        prediction = model.predict(input_df)
        probability = model.predict_proba(input_df)
        
        # Show results
        st.subheader('Prediction Results')
        income_class = '>50K' if prediction[0] == 1 else '≤50K'
        st.write(f'Predicted Income Class: {income_class}')
        st.write(f'Probability of >50K: {probability[0][1]:.2%}')
        
    except Exception as e:
        st.error(f'An error occurred: {str(e)}')
        st.error('Please make sure all inputs are valid and try again.')