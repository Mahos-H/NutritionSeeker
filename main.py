
import streamlit as st
import random
import os
from PIL import Image
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from model import NoviceNutriVision
from utils import load_model, predict 
import torch
from torchvision import transforms
st.set_page_config(page_title="NutriVision Inference", layout="centered")
torch.classes.__path__ = []
device = "cuda" if torch.cuda.is_available() else "cpu"

def setup_ui():
    
    st.markdown("""
        <style>
            body {
                background-color: #f0f8f0;
                color: #2d6a4f;
            }
            .stButton>button {
                background-color: #2d6a4f;
                color: white;
            }
            .stButton>button:hover {
                background-color: #1e4e37;
            }
            .stFileUploader>label {
                background-color: #a8dadc;
                color: #1d3557;
            }
            .stTextInput>label {
                color: #2d6a4f;
            }
            .stTitle {
                color: #2d6a4f;
            }
        </style>
    """, unsafe_allow_html=True)

    st.title("NutriVision Nutrition Estimator")
    st.write("Upload a food image and get the estimated nutritional details.")

    # Sidebar: Select Nutrition Source
    source_options = {
        "Food Nutrition (e.g., homemade foods)": "food_nutrition",
        "Fruits & Vegetables": "fv",
        "Fast Food": "fastfood"
    }
    selected_source_label = st.sidebar.selectbox("Select Nutrition Dataset", list(source_options.keys()))
    selected_source = source_options[selected_source_label]
    
    source_columns = {
        "food_nutrition": ["Caloric Value", "Fat", "Carbohydrates"],
        "fv": ["energy (kcal/kJ)", "water (g)", "protein (g)", "total fat (g)", "carbohydrates (g)", "fiber (g)", "sugars (g)", "calcium (mg)", "iron (mg)"],
        "fastfood": ["calories", "cal_fat", "total_fat", "sat_fat", "trans_fat", "cholesterol", "sodium", "total_carb"]
    }

    st.sidebar.markdown("### Expected Output Columns")
    st.sidebar.write(source_columns[selected_source])

    # Motivational Quote
    motivational_quotes = [
        "Eat Fresh, Stay Healthy!",
        "Healthy Life, Happy Life!",
        "Nourish Your Body, Nourish Your Soul!",
        "Fuel Your Body with the Right Nutrients!",
        "Stay Fit, Eat Well!"
    ]
    quote = random.choice(motivational_quotes)
    st.markdown(f"<h3 style='color:#1e4e37;'>💚 {quote} 💚</h3>", unsafe_allow_html=True)

    # Word Cloud
    health_terms = "nutrition health wellness fitness vitamins minerals healthy eating diet balance food calories fat carbohydrates protein fibers vitamins fruits vegetables stay fit clean eating habits lifestyle balance nutrition energy"
    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='Greens').generate(health_terms)
    
    plt.figure(figsize=(8, 4))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(plt)

    # Upload Image (No Webcam)
    uploaded_file = st.file_uploader("Upload an image (JPG, PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_container_width=True)
        except Exception as e:
            st.error(f"Error loading image: {e}")
            return

        # Image Preprocessing
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        image_tensor = transform(image).unsqueeze(0).to(device)

        # Load Model and Predict
        with st.spinner("Loading model and running inference..."):
            model, _ = load_model()
            pred_values, caption = predict(model, image_tensor, selected_source)

        # Display Results
        st.markdown("## Inference Results")
        st.write("Generated Caption:", caption)
        st.write("Predicted Nutritional Values:")

        columns = source_columns[selected_source]
        if len(pred_values) == len(columns):
            result_dict = {col: [round(val, 2)] for col, val in zip(columns, pred_values)}
            st.table(result_dict)
        else:
            st.write("Output vector:", pred_values)
    else:
        st.info("Please upload an image to get predictions.")

setup_ui()
