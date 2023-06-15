import streamlit as st

# Menu utama
menu = ["Data", "Preprocessing", "Modeling", "Implementasi"]
selected_menu = st.sidebar.selectbox("Menu", menu)

if selected_menu == "Data":
    st.header("Informasi Data")
    # Tambahkan penjelasan tentang data di sini

elif selected_menu == "Preprocessing":
    st.header("Preprocessing")
    preprocessing_option = st.selectbox("Pilihan Preprocessing", ["Min-Max Scaler", "Reduksi Dimensi"])

    if preprocessing_option == "Min-Max Scaler":
        st.subheader("Min-Max Scaler")
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
    # Tambahkan inputan untuk 3 input
    # Tambahkan kode untuk memuat data dan target, serta melakukan prediksi menggunakan model yang dipilih
    # Output hasil prediksi
