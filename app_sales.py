import pandas as pd
import numpy as np
import streamlit as st
from sklearn import *
import pickle

df = pickle.load(open('order_data.pkl', 'rb'))
best_model = pickle.load(open('sales_predict.pkl', 'rb'))

st.title('Sales Predictor of Amazing Mart')

st.header('Fill the details to Predict the Sales')

# Product - drop down
product = st.selectbox('Product', df['Product Name'].unique())
# Quantity - number input
quantity = st.number_input('Quantity', min_value=1, max_value=14)
# Discount - slider
discount = st.slider('Discount', min_value=0.0, max_value=0.6)
# Actual Discount - number input
actual_discount = st.number_input('Actual Discount', min_value=0, max_value=437)
# Profit - number input
profit = st.number_input('Profit', min_value=-93, max_value=701)
# Category - drop down
category = st.selectbox('Product Category', df['Category'].unique())
# Sub-Category - drop down
sub_category = st.selectbox('Sub Category', df['Sub-Category'].unique())
# Order Year - drop down
order_year = st.selectbox('Order Year', df['Order Year'].unique())
# Region - drop down
region = st.selectbox('Region', df['Region'].unique())
# Country - drop down
country = st.selectbox('Country', df['Country'].unique())
# State - drop down
state = st.selectbox('State', df['State'].unique())
# City - drop down
city = st.selectbox('City', df['City'].unique())
# Segment - drop down
segment = st.selectbox('Segment', df['Segment'].unique())
# Ship Year - drop down
ship_year = st.selectbox('Ship Year', df['Ship Year'].unique())
# Ship Mode - drop down
ship_mode = st.selectbox('Ship Mode', df['Ship Mode'].unique())
# Days to ship - number input
days_to_ship = st.number_input('Days to Ship', min_value=0, max_value=7)



if st.button('Predict Sales') :
    test_data = np.array([order_year, city, country, region, segment, ship_mode, state, ship_year,
                          days_to_ship,product, discount, actual_discount, profit, quantity, category, sub_category])
    test_data = test_data.reshape([1,16])
    
    st.success(best_model.predict(test_data)[0].round(2))