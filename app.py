import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load the model
loaded_model = load_model('dense_sigmoid_model.h5')

# Set target image size (you need to define IMAGE_SIZE)
IMAGE_SIZE = (200, 200)

# Function to make predictions
def predict_image_class(image_path):
    # Load and preprocess the image
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=IMAGE_SIZE)
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    # Predict the class
    classes = loaded_model.predict(x, batch_size=1)
    predicted_class_index = np.argmax(classes)

    class_labels = ['defect', 'longberry', 'peaberry', 'premium']

    if predicted_class_index < len(class_labels):
        predicted_label = class_labels[predicted_class_index]
        confidence = classes[0][predicted_class_index]
        return predicted_label, confidence
    else:
        return "Predicted class index is out of range.", 0.0
    
# Create a sidebar menu
st.sidebar.title("MENU")

selected_menu = st.sidebar.radio("", ["About", "Prediction"])

# About Page
# About Page
if selected_menu == "About":

    st.title("KLASIFIKASI BIJI KOPI")

    image = Image.open("C:/Users/ASUS/Desktop/BDC_SE_2023/BIJI_KOPI.jpg")
    st.image(image, caption='https://assets.kompasiana.com/items/album/ ')

        # Deskripsi
    st.write("Kopi merupakan salah satu komoditas perkebunan yang berperan besar dalam perekonomian dunia. Setiap jenis kopi memiliki bentuk dan tekstur yang berbeda-beda. Dalam kasus ini kopi diklasifikasikan kedalam 4 kelas yang berbeda  yaitu defect, longberry, peaberry dan premium")
    
    #DEFECT
    st.write("## Defect")
    image_defect = Image.open("DEFECT.jpg")
    image_defect = image_defect.resize((200, 200))  # Ubah ukuran gambar ke 200x200 piksel
    col1, col2 = st.columns(2)
    with col1:
        st.image(image_defect, use_column_width=True)
    with col2:
        st.write("Defect adalah salah satu varietas kopi yang memiliki cacat atau ketidaksempurnaan dalam bentuk biji kopi. Cacat ini bisa berupa cacat fisik seperti retak, pecah, atau cacat lainnya yang membuat biji kopi menjadi tidak sempurna. Cacat bisa disebabkan oleh berbagai faktor seperti pengolahan yang tidak benar atau masalah dalam proses pertumbuhan biji kopi. Ciri-ciri Biji kopi dalam varietas Defect seringkali memiliki penampilan fisik yang tidak sempurna, dengan cacat-cacat yang terlihat seperti retak, pecah, atau bentuk yang tidak biasa. Rasa kopi dari varietas ini mungkin kurang kualitasnya karena cacat-cacat tersebut.")

    # Longberry
    st.write("## Longberry")
    image_longberry = Image.open("LONGBERRY.jpg")
    image_longberry = image_longberry.resize((200, 200)) 
    col1, col2 = st.columns(2)
    with col1:
        st.image(image_longberry, use_column_width=True)
    with col2:
        st.write("Longberry adalah varietas kopi yang dikenal karena bentuk biji kopi yang lebih panjang dan sempurna. Biji kopi Longberry memiliki bentuk yang lebih oval dan panjang dibandingkan dengan varietas kopi lainnya. Varietas ini terkenal karena rasa kopi yang khas dan unik. Ciri-ciri Biji kopi Longberry memiliki bentuk yang panjang dan sempurna, dengan warna yang seragam. Rasa kopi dari varietas ini seringkali lebih kuat dan memiliki karakteristik rasa yang unik.")
        
    # Peaberry    
    st.write("## Peaberry")
    image_peaberry = Image.open("PEABERRY.jpg")
    image_peaberry = image_peaberry.resize((200, 200)) 
    col1, col2 = st.columns(2)
    with col1:
        st.image(image_peaberry, use_column_width=True)
    with col2:
        st.write("Peaberry adalah varietas kopi yang memiliki biji kopi yang lebih kecil dan bulat dibandingkan dengan biji kopi konvensional. Varietas ini terkenal karena setiap biji kopi hanya memiliki satu biji dalam buah kopi, bukan dua. Ini membuat biji Peaberry memiliki bentuk yang bulat dan padat. Ciri-ciri Biji kopi Peaberry memiliki bentuk yang bulat dan biasanya lebih kecil daripada biji kopi konvensional. Rasa kopi dari varietas ini seringkali dianggap memiliki karakteristik yang lebih khas dan unik.")

    # Premium    
    st.write("## Premium")
    image_premium = Image.open("PREMIUM.jpg")
    image_premium = image_premium.resize((200, 200)) 
    col1, col2 = st.columns(2)
    with col1:
        st.image(image_premium, use_column_width=True)
    with col2:
        st.write("Premium adalah varietas kopi yang dianggap memiliki kualitas yang sangat tinggi. Bijinya dipilih dengan cermat dan diproses dengan hati-hati untuk menghasilkan kopi dengan rasa yang superior. Premium coffee sering berasal dari tanaman kopi yang tumbuh di kondisi lingkungan yang optimal dan diproses dengan metode khusus. Ciri-ciri: Biji kopi Premium seringkali memiliki bentuk dan penampilan yang sempurna. Rasa kopi dari varietas ini dianggap sangat baik, dengan karakteristik rasa yang khas dan bervariasi tergantung pada asal-usul dan metode pemrosesan.")
    

# Prediction Page
if selected_menu == "Prediction":
    st.title("PREDIKSI KELAS BIJI KOPI")

    # Upload an image using Streamlit
    uploaded_image = st.file_uploader("Upload gambar", type=["jpg", "png", "jpeg"])
    
    if uploaded_image is not None:
        st.image(uploaded_image, use_column_width=True)
        
        if st.button("Predict"):
            with st.spinner("Predicting..."):
                predicted_label, confidence = predict_image_class(uploaded_image)
                st.success(f"Predicted class: {predicted_label}")
                st.info(f"Confidence: {confidence:.2f}")
