import streamlit as st
import sklearn

# Write the version of scikit-learn to the Streamlit app
st.write(f"Scikit-learn version: {sklearn.__version__}")
