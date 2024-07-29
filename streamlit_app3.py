import streamlit as st
import pandas as pd
import joblib
import os

def input_swimmer_data_streamlit(num_splits):
    data = []
    st.write("Swimmer 1")
    swimmer_data = []
    for split in range(1, num_splits + 1):
        key = f"split_{split}"
        split_time = st.number_input(f"Enter split time for split {split}:", min_value=0.0, step=0.01, key=key)
        swimmer_data.append(split_time)
    data.append(swimmer_data)

    df = pd.DataFrame(data, columns=[f"Split {i+1}" for i in range(num_splits)])
    df.insert(0, "Swimmer", [1])

    column_names = [f"{50 * (i + 1)}m" for i in range(num_splits)]
    column_names.insert(0, "Swimmer")
    df.columns = column_names

    return df

def manipulate_data(df):
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

    selected_columns = df.columns[1:31]
    for i, column in enumerate(selected_columns):
        new_column_name = f'{column}_speed'
        df[new_column_name] = 50 / df[column]

    df['mean_speed'] = df.iloc[:, 43:73].mean(axis=1)
    df['std'] = df.iloc[:, 43:73].std(axis=1)
    df['CV%'] = (df.iloc[:, 43:73].std(axis=1) / df.iloc[:, 65:87].mean(axis=1)) * 100

    df['mean_time'] = df.iloc[:, 1:31].mean(axis=1)
    df['std_time'] = df.iloc[:, 1:31].std(axis=1)
    df['CV_time'] = (df.iloc[:, 1:31].std(axis=1) / df.iloc[:, 13:43].mean(axis=1)) * 100

    return df

def main():
    st.title("Swimmer Split Time Input - author: Tiago Russomanno")

    num_splits = 30

    df = input_swimmer_data_streamlit(num_splits)

    if st.button("Submit"):
        df = manipulate_data(df)

        #st.write("Swimmer Split Times DataFrame with Calculated Variables:")
        #st.write(df)

        model_path = 'saved_rf_model.pkl'  # Update this path as needed
        model = joblib.load(model_path)

        selected_features_new = ["mean_time", "1000T", "mean_speed", "800m_speed", "850m", "950m", "850m_speed", "800m", "950m_speed", "1250T"]

        input_data = df[selected_features_new]

        predictions = model.predict(input_data)

        total_final_time = predictions.sum()

        # Convert total final time to minutes and seconds
        minutes = int(total_final_time // 60)
        seconds = total_final_time % 60

        #st.write("Prediction Results:")
        #st.write(predictions)

        st.write("Total Final Time Based on Model Predictions:")
        st.write(f"{minutes} minutes and {seconds:.2f} seconds")

if __name__ == "__main__":
    main()