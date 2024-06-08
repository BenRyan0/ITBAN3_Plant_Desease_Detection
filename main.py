import streamlit as st
import tensorflow as tf
import numpy as np
import base64

# Read the image file for the favicon
file_path = "image/logo_icon.png"  # Replace with your image file path
with open(file_path, "rb") as f:
    img_bytes = f.read()

# Encode image to base64
encoded_img = base64.b64encode(img_bytes).decode()

# Set the page configuration
st.set_page_config(
    page_title="LeafMD",  # Title of the browser tab
    page_icon=f"data:image/png;base64,{encoded_img}",  # Favicon as base64 encoded image
    layout="centered",  # Layout can be "centered" or "wide"
)

# Inject custom CSS
st.markdown("""
    <style>
   @import url('https://fonts.googleapis.com/css2?family=Poppins:ital,wght@0,100;0,200;0,300;0,400;0,500;0,600;0,700;0,800;0,900;1,100;1,200;1,300;1,400;1,500;1,600;1,700;1,800;1,900&display=swap');

            
    
    *{
        font-family: "Poppins", sans-serif;
    }
     .stButton>button {
        background-color: #02A367;
        border: 2px solid #02A367;
        color: white;
        padding: 15px 32px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        font-weight: 700;
        margin: 3px 2px;
        cursor: pointer;
        border-radius: 10px;
        transition: .3s ease-in-out; 
    }
    .stButton>button:hover{
        border: 2px solid #02A367;
        background-color: #fff;
        color: #02A367

    }
    .stButton>button:active{
        background-color: #02A367;
        border: 2px solid #02A367;
        color: white;
    
    }
  
            
   .stFileUpload button {
        background-color: #4CAF50 !important; /* Green background */
        color: white !important;
        padding: 10px !important;
        border-radius: 5px !important;
        font-weight: bold !important;
        cursor: pointer !important;
    }
    .stFileUpload button:hover {
        background-color: #45a049 !important; /* Darker green background on hover */
    }
    </style>
    """, unsafe_allow_html=True)

# TensorFlow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model_IT3A.h5")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # Return index of max element

# Sidebar
image_path = "image/logo_full.png"
st.sidebar.image(image_path, width=200)
app_mode = st.sidebar.selectbox("Choose an option:", ["Home", "About", "Disease Recognition"], label_visibility="hidden")

# Main Page
if app_mode == "Home":
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "image/Main_BG.jpg"
    st.image(image_path, use_column_width=True)
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍
    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)

elif app_mode == "About":
    st.header("About")
    st.markdown("""
    #### About Dataset
    This dataset is recreated using offline augmentation from the original dataset. The original dataset can be found on this github repo.
    This dataset consists of about 87K RGB images of healthy and diseased crop leaves which is categorized into 38 different classes. The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure.
    A new directory containing 33 test images is created later for prediction purposes.
    #### Content
    1. train (70,295 images)
    2. test (33 images)
    3. validation (17,572 images)
    """)
    st.markdown("[Dataset Used In Training](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)")
    st.markdown("[Code Github Repository](https://github.com/BenRyan0/ITBAN3_Plant_Desease_Detection.git)")

    st.header("Our Team")
    
    team_members = [
        {"name": "Joanna Angel C. Lugasan", "image": "image/team/Angel.png"},
        {"name": "Ma. Ella Mae Torrejos ",  "image": "image/team/Angel.png"},
        {"name": "Mark John C. Granada",  "image": "image/team/Mark.png"},
        {"name": "Bonjovie A. Belbelone",  "image": "image/team/Bon.png"},
        {"name": "Ryan James B. Baya",  "image": "image/team/Baya.png"},
        {"name": "Ben Ryan A. Rinconada", "image": "image/team/Ben.png"},
    ]

    cols = st.columns(3)  # Adjust the number of columns based on the number of team members

    for idx, member in enumerate(team_members):
        with cols[idx % 3]:  # Loop through the columns
            st.image(member["image"], width=200)
            st.subheader(member["name"])
            st.divider()
        

elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:" )
    if st.button("Show Image", type="primary"):
        if test_image is not None:
            st.image(test_image, use_column_width=True)
        else:
            st.warning("Please upload an image before clicking 'Show Image'.")
    if st.button("Predict",  type="primary"):
        if test_image is not None:
            with st.spinner('Predicting...'):
                result_index = model_prediction(test_image)
                class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                              'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                              'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                              'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                              'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                              'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                              'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                              'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                              'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                              'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                              'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                              'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                              'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                              'Tomato___healthy']
            st.success(f"Model is Predicting it's a {class_name[result_index]}")
        else:
            st.warning("Please upload an image before clicking 'Predict'.")
