import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error

def split_sequence(sequence, n_steps):
    X, y = [], []
    for i in range(len(sequence)-n_steps):
        X.append(sequence[i:i+n_steps])
        y.append(sequence[i+n_steps])
    return np.array(X), np.array(y)

def main():
    st.title("Aplikasi Analisis Keuangan PT. KAI")
    st.subheader("Data diambil dari Yahoo Financial")

    # Membaca dataset
    df_data = pd.read_csv('KAI.csv')
    st.write("Data Awal:")
    st.dataframe(df_data.head(7))

    # Menampilkan ukuran dataset
    st.write("Ukuran Dataset:")
    st.write(df_data.shape)

    # Memproses data
    df_high = df_data['High']
    n_steps = 2
    X, y = split_sequence(df_high, n_steps)

    # Membuat data frame untuk X dan y
    df_X = pd.DataFrame(X, columns=['t-'+str(i) for i in range(n_steps-1, -1, -1)])
    df_y = pd.DataFrame(y, columns=['t+1 (prediction)'])

    # Menggabungkan df_X dan df_y
    df = pd.concat([df_X, df_y], axis=1)
    st.write("Data Setelah Diproses:")
    st.dataframe(df.head(3))

    # Melakukan normalisasi menggunakan MinMaxScaler
    scaler = MinMaxScaler()
    X_norm = scaler.fit_transform(df_X)
    y_norm = scaler.fit_transform(df_y)

    # Membagi data menjadi data latih dan data uji
    X_train, X_test, y_train, y_test = train_test_split(X_norm, y_norm, test_size=0.2, random_state=0)

    # Menggunakan model K-Nearest Neighbors
    model_knn = KNeighborsRegressor(n_neighbors=3)
    model_knn.fit(X_train, y_train)
    y_pred_knn = model_knn.predict(X_test)
    r_squared_knn = r2_score(y_test, y_pred_knn)

    # Menggunakan model Linear Regression
    model_lr = LinearRegression()
    model_lr.fit(X_train, y_train)
    y_pred_lr = model_lr.predict(X_test)
    r_squared_lr = r2_score(y_test, y_pred_lr)

    # Menggunakan model Decision Tree
    model_dt = DecisionTreeRegressor()
    model_dt.fit(X_train, y_train)
    y_pred_dt = model_dt.predict(X_test)
    r_squared_dt = r2_score(y_test, y_pred_dt)

    # Menampilkan hasil
    st.subheader("Hasil Analisis:")
    st.write("R-squared (K-Nearest Neighbors):", r_squared_knn)
    st.write("R-squared (Linear Regression):", r_squared_lr)
    st.write("R-squared (Decision Tree):", r_squared_dt)

    st.subheader("Metrik Evaluasi:")
    mse_lr = mean_squared_error(y_test, y_pred_lr)
    mape_lr = mean_absolute_percentage_error(y_test, y_pred_lr)
    st.write("MSE (Linear Regression):", mse_lr)
    st.write("MAPE (Linear Regression):", mape_lr)

    st.write("Dimensi y_test:", y_test.shape)
    st.write("Dimensi y_pred (Linear Regression):", y_pred_lr.shape)

    # Menampilkan dataframe y_test dan y_pred
    df_y_test = pd.DataFrame(scaler.inverse_transform(y_test), columns=['y_test'])
    df_y_pred = pd.DataFrame(scaler.inverse_transform(y_pred_lr.reshape(-1, 1)), columns=['y_pred'])
    df_hasil = pd.concat([df_y_test, df_y_pred], axis=1)
    st.write("Dataframe y_test dan y_pred (Linear Regression):")
    st.dataframe(df_hasil)

if __name__ == '__main__':
    main()
