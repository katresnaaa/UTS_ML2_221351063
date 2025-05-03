import streamlit as st
import tensorflow as tf
import numpy as np
import joblib

# Load scaler
scaler = joblib.load('scaler.pkl')

# Load model TFLite
interpreter = tf.lite.Interpreter(model_path="Jamur.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Judul Aplikasi
st.title("Mushroom Classification")
st.write("Masukkan informasi tentang jamur untuk mengetahui apakah jamur tersebut beracun atau tidak.")

# Form input pengguna
cap_shape = st.selectbox("Bentuk Cap Jamur", [0, 1, 2, 3])  # Nilai sudah encoded
cap_surface = st.selectbox("Permukaan Cap Jamur", [0, 1, 2, 3])  # Nilai sudah encoded
cap_color = st.selectbox("Warna Cap Jamur", [0, 1, 2, 3, 4])  # Nilai sudah encoded
bruises = st.selectbox("Apakah Jamur Berdarah (Bruises)?", [0, 1])  # Nilai sudah encoded
odor = st.selectbox("Bau Jamur", [0, 1, 2, 3, 4])  # Nilai sudah encoded
gill_attachment = st.selectbox("Pelekatan Gill", [0, 1])  # Nilai sudah encoded
gill_spacing = st.selectbox("Jarak Gill", [0, 1])  # Nilai sudah encoded
gill_size = st.selectbox("Ukuran Gill", [0, 1])  # Nilai sudah encoded
gill_color = st.selectbox("Warna Gill", [0, 1, 2, 3, 4, 5, 6])  # Nilai sudah encoded
stalk_shape = st.selectbox("Bentuk Stalk Jamur", [0, 1])  # Nilai sudah encoded
stalk_root = st.selectbox("Akar Stalk Jamur", [0, 1, 2, 3, 4, 5, 6])  # Nilai sudah encoded
stalk_surface_above_ring = st.selectbox("Permukaan Stalk di Atas Cincin", [0, 1, 2, 3])  # Nilai sudah encoded
stalk_surface_below_ring = st.selectbox("Permukaan Stalk di Bawah Cincin", [0, 1, 2, 3])  # Nilai sudah encoded
stalk_color_above_ring = st.selectbox("Warna Stalk di Atas Cincin", [0, 1, 2, 3, 4])  # Nilai sudah encoded
stalk_color_below_ring = st.selectbox("Warna Stalk di Bawah Cincin", [0, 1, 2, 3, 4])  # Nilai sudah encoded
veil_color = st.selectbox("Warna Selubung", [0, 1, 2])  # Nilai sudah encoded
ring_number = st.selectbox("Jumlah Cincin", [0, 1, 2])  # Nilai sudah encoded
ring_type = st.selectbox("Jenis Cincin", [0, 1, 2, 3, 4, 5, 6])  # Nilai sudah encoded
spore_print_color = st.selectbox("Warna Cetakan Spora", [0, 1, 2, 3, 4, 5, 6])  # Nilai sudah encoded
population = st.selectbox("Populasi Jamur", [0, 1, 2, 3, 4])  # Nilai sudah encoded
habitat = st.selectbox("Habitat Jamur", [0, 1, 2, 3, 4])  # Nilai sudah encoded

# Mengonversi input ke array sesuai urutan yang diminta
input_data = np.array([[cap_shape, cap_surface, cap_color, bruises, odor, gill_attachment, gill_spacing, gill_size,
                        gill_color, stalk_shape, stalk_root, stalk_surface_above_ring, stalk_surface_below_ring,
                        stalk_color_above_ring, stalk_color_below_ring, veil_color, ring_number, ring_type,
                        spore_print_color, population, habitat]])

# Proses scaling (menggunakan scaler yang telah disimpan)
input_scaled = scaler.transform(input_data).astype(np.float32)

# Prediksi ketika tombol ditekan
if st.button("Prediksi Jamur"):
    # Set tensor untuk input
    interpreter.set_tensor(input_details[0]['index'], input_scaled)
    interpreter.invoke()

    # Ambil hasil prediksi
    prediction = interpreter.get_tensor(output_details[0]['index'])

    # Mengubah hasil prediksi menjadi label
    predicted_label = np.argmax(prediction)
    mushroom_type = "Beracun" if predicted_label == 1 else "Tidak Beracun"

    st.success(f"Jenis jamur: **{mushroom_type}**")
