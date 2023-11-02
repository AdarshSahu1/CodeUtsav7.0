import streamlit as st
import pickle
import numpy as np

# Load the trained linear regression model
with open('C:/Users/HP/Desktop/NIRFFINAL/linear_regression_model.pkl', 'rb') as model_file:
    linear_regression_model = pickle.load(model_file)

st.markdown(
    """
    <style>
    /* Custom styles */
    body {
        font-family: 'Arial', sans-serif; /* Change the font */
        background-color: #f7f7f7; /* Background color */
        color: #333; /* Text color */
    }
    .stButton button {
        background-color: #007bff; /* Button background color */
        color: #fff; /* Button text color */
        border-radius: 5px;
        font-weight: bold;
    }
    .stTextInput input {
        background-color: #fff; /* Input field background color */
        border: 1px solid #ccc; /* Input field border */
        border-radius: 5px;
        font-size: 16px; /* Input field font size */
    }
    .stNumberInput input[type=number]::-webkit-inner-spin-button,
    .stNumberInput input[type=number]::-webkit-outer-spin-button {
        -webkit-appearance: none;
        margin: 0;
    }
    .stNumberInput input[type=number] {
        font-size: 16px; /* Number input font size */
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def predict_college_rank(user_data, model):
    # Prepare the input data for prediction
    user_data_list = [user_data[key] for key in user_data]
    user_data_array = np.array(user_data_list).reshape(1, -1)

    # Make the prediction
    prediction = model.predict(user_data_array)

    # Round the prediction to the nearest integer
    rank = int(round(prediction[0]))

    return rank


def main():
    # Set page title
    st.title("College Rank Predictor")

    # Style input widgets
    st.sidebar.subheader("Input Parameters")

    user_data = {
        'Teaching, Learning, and Resources': st.sidebar.number_input('Teaching, Learning, and Resources', min_value=0, max_value=100, step=1),
        'Research and Professional Practice': st.sidebar.number_input('Research and Professional Practice', min_value=0, max_value=100, step=1),
        'Graduation Outcome': st.sidebar.number_input('Graduation Outcome', min_value=0, max_value=100, step=1),
        'Outreach and Inclusivity': st.sidebar.number_input('Outreach and Inclusivity', min_value=0, max_value=100, step=1),
        'Perception Score': st.sidebar.number_input('Perception Score', min_value=0, max_value=100, step=1),

    }

    # Predict rank
    if st.sidebar.button('Predict'):
        result = predict_college_rank(user_data, linear_regression_model)
        st.subheader("Predicted College Rank")
        st.write(f"Rank: {result}")


if __name__ == "__main__":
    main()
