import streamlit as st
from numpy import array
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier

# Fungsi untuk melakukan Min-Max Scaling
def min_max_scaling(data):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data

# Fungsi untuk melakukan reduksi dimensi menggunakan PCA
def dimension_reduction(data):
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(data)
    return reduced_data

# Fungsi untuk melakukan prediksi menggunakan KNN
def knn_predict(data, target, new_data):
    knn = KNeighborsClassifier()
    knn.fit(data, target)
    prediction = knn.predict(new_data)
    return prediction

# Fungsi untuk melakukan prediksi menggunakan Regresi Linear
def linear_regression_predict(data, target, new_data):
    linear_regression = LinearRegression()
    linear_regression.fit(data, target)
    prediction = linear_regression.predict(new_data)
    return prediction

# Fungsi untuk melakukan prediksi menggunakan Decision Tree
def decision_tree_predict(data, target, new_data):
    decision_tree = DecisionTreeClassifier()
    decision_tree.fit(data, target)
    prediction = decision_tree.predict(new_data)
    return prediction

# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
    # find the end of this pattern
        end_ix = i + n_steps
    # check if we are beyond the sequence
        if end_ix > len(sequence)-1:
            break
    # gather input and output parts of the pattern
        # print(i, end_ix)
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

# Menu utama
st.title("Prediksi Keuangan PT.KAI")

menu = ["Data", "Preprocessing", "Modeling", "Implementasi"]
selected_menu = st.sidebar.selectbox("Menu", menu)

if selected_menu == "Data":
    st.header("Aplikasi ini dibuat untuk memprediksi harga tertinggi saham PT.KAI")
    st.header("untuk data, kami mendapatkan data tersebut dari yahoo financial")
    st.header("untuk tipe data yang ada pada data yang kami ambil adalah numerik")
    st.header("data yang kami dapat merupakan data saham PT.KAI")
    data_file = st.file_uploader("Inputkan data", type=["csv"])
    if data_file is not None:
        data = pd.read_csv(data_file)
        df_high=data['High']
        # transform univariate time series to supervised learning problem
        n_steps = 2
        X, y = split_sequence(df_high, n_steps)
        df_X = pd.DataFrame(X, columns=['t-'+str(i) for i in range(n_steps-1, -1,-1)])
        df_y = pd.DataFrame(y, columns=['t+1 (prediction)'])

# concat df_X and df_y
        df = pd.concat([df_X, df_y], axis=1)
        st.write(data)
        st.session_state.data = data

elif selected_menu == "Preprocessing":
    st.header("Preprocessing")
    preprocessing_option = st.selectbox("Pilihan Preprocessing", ["Min-Max Scaler", "Reduksi Dimensi"])

    if preprocessing_option == "Min-Max Scaler":
        st.subheader("Min-Max Scaler")
        if "data" in st.session_state:
            data = st.session_state.data  # Akses data dari session state
            scaled_data = min_max_scaling(data)
            st.write(scaled_data)
        # Tambahkan kode untuk memuat data dan melakukan Min-Max Scaler
        # Output hasil preprocessing

    elif preprocessing_option == "Reduksi Dimensi":
        st.subheader("Reduksi Dimensi")
        # Tambahkan kode untuk memuat data dan melakukan reduksi dimensi menggunakan PCA
        # Output hasil preprocessing

elif selected_menu == "Modeling":
    st.header("Modeling")
    model_option = st.selectbox("Pilihan Model", ["KNN", "Regresi Linear", "Decision Tree"])

    if model_option == "KNN":
        st.subheader("K-Nearest Neighbors (KNN)")
        # Tambahkan kode untuk memuat data dan target, serta inputan untuk prediksi menggunakan KNN
        # Output hasil prediksi

    elif model_option == "Regresi Linear":
        st.subheader("Regresi Linear")
        # Tambahkan kode untuk memuat data dan target, serta inputan untuk prediksi menggunakan Regresi Linear
        # Output hasil prediksi

    elif model_option == "Decision Tree":
        st.subheader("Decision Tree")
        # Tambahkan kode untuk memuat data dan target, serta inputan untuk prediksi menggunakan Decision Tree
        # Output hasil prediksi

elif selected_menu == "Implementasi":
    st.header("Implementasi")
    # Input untuk 3 fitur
    feature1 = st.number_input("Fitur 1")
    feature2 = st.number_input("Fitur 2")
    feature3 = st.number_input("Fitur 3")

    # Tambahkan kode untuk memuat data dan melakukan prediksi menggunakan model yang dipilih
    if st.button("Prediksi"):
        # Simulasikan data dan target (ganti dengan data dan target yang sesuai)
        data = pd.DataFrame([[100], [4, 5, 6], [7, 8, 9]])
        target = pd.Series([0, 1, 0])

        # Proses data yang dimuat dan inputan menggunakan model yang dipilih
        # Misalnya, menggunakan KNN untuk prediksi
        new_data = pd.DataFrame([[feature1, feature2, feature3]])
        prediction = knn_predict(data, target, new_data)

        # Output hasil prediksi
        st.subheader("Hasil Prediksi")
        st.write(prediction)