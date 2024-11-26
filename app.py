import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder


# Set background
def set_bg(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{image_url}");
            background-size: cover;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
set_bg("https://img.freepik.com/premium-photo/abstract-glowing-orange-graph-dark-background-suitable-technology-business-finance-concepts_14117-205013.jpg")

# Set title for the Streamlit app
st.title("Financial Goal Prediction App")

# Sidebar for navigation
st.sidebar.title("Navigation")
choice = st.sidebar.radio("Choose an option:", ["Overview", "Prediction", "Visualize Data"])

# Add App Details and Credits in Sidebar
st.sidebar.title("About the App")
st.sidebar.info("""
**Financial Goal Prediction App** is a powerful tool designed to:
- Predict financial goals based on various inputs like expenses, income, and savings.
- Provide insightful visualizations of financial data.
- Help users understand and manage their financial goals effectively.

**Features**:
- Data Overview and Statistics
- Machine Learning-based Predictions
- Interactive Visualizations

Created by: **Darshanikanta**  
© 2024 All Rights Reserved
""")

# Load the trained model and scaler
@st.cache_resource
def load_model_and_scaler():
    with open('financial goal.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    return model, scaler

model, scaler = load_model_and_scaler()

# Load the dataset
@st.cache_data
def load_data():
    file_path = "family_financial_and_transactions_data.xlsx"
    return pd.read_excel(file_path)

data = load_data()

if choice == "Overview":
    # Display dataset info
    st.header("Dataset Overview")
    st.write("Here's a preview of the dataset:")
    st.dataframe(data.head())

    # Show dataset statistics
    st.subheader("Dataset Statistics")
    st.write(data.describe())

elif choice == "Prediction":
    st.header("Make Predictions")
    
    # Input fields for user to enter data
    st.write("Enter the required input features below:")

    # Select category (categorical input)
    category = st.selectbox("Select Category:", sorted(data['Category'].unique()))
    
    # Numeric inputs
    amount = st.number_input("Enter Amount Spent (e.g., 200):", step=1.0)
    income = st.number_input("Enter Monthly Income (e.g., 50000):", step=1.0)
    saving = st.number_input("Enter Savings (e.g., 10000):", step=1.0)
    monthly_expense = st.number_input("Enter Monthly Expense (e.g., 20000):", step=1.0)
    loan_payment = st.number_input("Enter Loan Payment (e.g., 5000):", step=1.0)
    credit_card = st.number_input("Enter Credit Card Bill (e.g., 15000):", step=1.0)
    dependents = st.number_input("Enter Number of Dependents (e.g., 2):", step=1)
    
    # Label Encoding for Category
    label_encoder = LabelEncoder()
    label_encoder.fit(data['Category'])
    category_encoded = label_encoder.transform([category])[0]  # Transform the category input
    
    # Combine all inputs into a single array
    input_array = np.array([[category_encoded, amount, income, saving, monthly_expense, loan_payment, credit_card, dependents]])
    input_scaled = scaler.transform(input_array)  # Scale the inputs

    # Make predictions
    if st.button("Predict"):
        prediction = model.predict(input_scaled)
        st.success(f"The predicted financial goal is: {prediction[0]:.2f}%")

elif choice == "Visualize Data":
    st.header("Data Visualization")

    # Select numerical columns for visualization
    numerical_columns = data.select_dtypes(include=['number']).columns.tolist()
    
    # Prompt user to select columns for visualization
    columns_to_plot = st.multiselect("Choose numerical columns to visualize:", numerical_columns)
    
    if len(columns_to_plot) > 1:
        st.line_chart(data[columns_to_plot])
    elif len(columns_to_plot) == 1:
        st.warning("Please select at least two columns for visualization.")
    else:
        st.warning("No columns selected for visualization. Please select at least two numerical columns.")
