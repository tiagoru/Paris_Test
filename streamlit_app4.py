import streamlit as st
import pandas as pd
import joblib

def input_swimmer_data_streamlit(num_splits):
    data = {}
    for i in range(1, num_splits + 1):
        data[f'Split_{i}'] = st.number_input(f'Split {i}', min_value=0.0, format="%.2f")
    df = pd.DataFrame([data])
    return df

def manipulate_data(df):
    # Ensure the DataFrame has at least 31 columns
    if df.shape[1] < 31:
        raise ValueError("DataFrame must have at least 31 columns")

    # Fill missing values with 0 to avoid issues during summation
    df = df.fillna(0)

    df['FH'] = df.iloc[:, 1:16].sum(axis=1)
    df['SH'] = df.iloc[:, 16:31].sum(axis=1)
    df['First'] = df.iloc[:, 1:11].sum(axis=1)
    df['Second'] = df.iloc[:, 11:21].sum(axis=1)
    df['Third'] = df.iloc[:, 21:31].sum(axis=1)
    df['250T'] = df.iloc[:, 1:6].sum(axis=1)
    df['500T'] = df.iloc[:, 6:11].sum(axis=1)
    df['750T'] = df.iloc[:, 11:16].sum(axis=1)
    df['1000T'] = df.iloc[:, 16:21].sum(axis=1)
    df['1250T'] = df.iloc[:, 21:26].sum(axis=1)
    df['1500T'] = df.iloc[:, 26:31].sum(axis=1)
    df['Total'] = df.iloc[:, 1:31].sum(axis=1)

    return df

def load_model(file_path):
    try:
        model = joblib.load(file_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def main():
    st.title("Swimmer Data Input")
    num_splits = 31  # Example value, adjust as needed
    df = input_swimmer_data_streamlit(num_splits)

    if st.button("Submit"):
        try:
            df = manipulate_data(df)

            model_path = 'saved_rf_model.pkl'  # Update this path as needed
            model = load_model(model_path)

            if model is not None:
                selected_features_new = ["mean_time", "1000T", "mean_speed", "800m_speed", "850m", "950m", "850m_speed", "800m", "950m_speed", "1250T"]

                input_data = df[selected_features_new]

                predictions = model.predict(input_data)

                total_final_time = predictions.sum()
                st.write(f"Total Final Time: {total_final_time}")
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()