# If wanna run the app, type in terminal "streamlit run app.py"
from importlib import reload
import streamlit as st
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
from streamlit import page_link
from torchvision import models
import requests
import random

@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=True)
    model.eval()
    return model

model = load_model()


@st.cache_data
def load_labels():
    LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    return requests.get(LABELS_URL).text.strip().split("\n")

labels = load_labels()

# Emoji- for kids lmao
emojis = {
    'healthy': '🥦💪🍎',
    'unhealthy': '🍟🍩😬',
    'neutral': '🍽️🙂',
    'unidentified': '🤔❓'
}


tips_and_fun_facts = [
    "🌈 Eat a rainbow of fruits and veggies every day!",
    "🥤 Drink water, not sugary drinks!",
    "🍚 Make half your plate veggies!",
    "🥕 Crunchy veggies make great snacks!",
    "🍌Bananas are technically berries, but strawberries aren't🤯",
    "🍓 Strawberries are the only fruit with seeds on the outside.",
    "🥕 Carrots were originally purple, not orange!",
    "🍍 Pineapples take almost two years to grow!",
    "🌽 Corn is a type of grass — and each kernel is a seed.",
    "🍫 Chocolate was once used as money by the Aztecs!",
    "🥦 Broccoli contains more protein than steak (per calorie)!",
    "🥚 Brown eggs and white eggs are nutritionally the same.",
    "🍯 Honey never spoils — archaeologists have found 3000-year-old honey still good to eat!",
    "🍏 Apples float because they’re 25% air.",
    "🍊 Oranges aren’t always orange — in warm countries they can stay green!",
    "🥬 Lettuce is a member of the sunflower family.",
    "🍞 Bread was once used to erase pencil marks before rubber erasers were invented!",
    "🥒 Pickles are cucumbers soaked in salty water or vinegar.",
    "🥜 Peanuts aren’t actually nuts — they’re legumes like beans!"
 ]

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Patrick+Hand&display=swap');

    .fun-header {
        font-family: 'Patrick Hand', cursive;
        color: #1b4332;
        font-size: 2.4em;
        text-align: center;
        margin-top: 10px;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)
# SnackSnap App
st.title("SnackSnap🥑")
st.markdown("""
    <div style="overflow: hidden;">
        <div style="float: right; margin-right: 300px;">
            <img src="https://media2.giphy.com/media/v1.Y2lkPTc5MGI3NjExMXR0b3VoMWVzNzY1Mzh1b2RjeDlkMmJ0bWJpd2FkNzZmNWVlZTJqdSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9cw/I0XDGG3idZ7HdXernP/giphy.gif" width="120">
        </div>
    </div>
""", unsafe_allow_html=True)
st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Comic+Neue:wght@700&display=swap" rel="stylesheet">
    <style>
    html, body, [class*="st-"] {
        font-family: 'Comic Neue', cursive;
    }

    h1 {
        font-size: 2.8em !important;
        color: #1b4332;
        text-shadow: 1px 1px #d8f3dc;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='fun-header'>🍽️ What Are You Eating Today?</div>", unsafe_allow_html=True)
st.subheader("📸Snap me a photo, and I’ll tell you how healthy it is!")
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Chewy&display=swap');

    div.stButton > button {
        font-family: 'Chewy', cursive;
        font-size: 1.8rem;
        padding: 1rem 2.5rem;
        border: none;
        border-radius: 50px;
        background: linear-gradient(135deg, #fcd34d, #fbbf24);
        color: #4b0082;
        box-shadow: 4px 6px 12px rgba(0,0,0,0.2);
        animation: bounce 1.6s infinite;
        transition: transform 0.2s ease-in-out;
        display: block;
        margin: 2rem auto;
    }

    div.stButton > button:hover {
        background: linear-gradient(135deg, #fbbf24, #fcd34d);
        transform: scale(1.07);
        cursor: pointer;
    }

    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-5px); }
    }
    </style>
""", unsafe_allow_html=True)
# Initialize session state variables if not already set
if "show_camera" not in st.session_state:
    st.session_state["show_camera"] = False

# Simple centered Streamlit-native button
if st.button("📸 Tap to Snap a Yummy Pic!"):
    st.session_state.show_camera = True

# Show camera input only after tapping the button
img_data = None

if img_data:
    image = Image.open(img_data).convert('RGB')
    st.image(image, caption="Here's your snap!", use_column_width=True)

# Only show camera input after button is pressed
if st.session_state.show_camera:
    img_data = st.camera_input("Tap below to take a photo")

# If no camera image, offer file upload
if img_data is None:
    uploaded_file = st.file_uploader("Or pick a photo of your liking 😄", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img_data = uploaded_file

# If any image is provided
if img_data:
    image = Image.open(img_data).convert('RGB')
    st.image(image, caption="Here's your snap!", use_column_width=True)

    # Preprocessing
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        _, predicted_idx = torch.max(output, 1)
        food_name = labels[predicted_idx.item()]


    # Health analysis
    food_name = food_name.lower()
    healthy_foods = ["apple", "banana", "orange", "pear", "grapes", "blueberries", "strawberries", "kiwi", "peach", "pineapple",
    "broccoli", "spinach", "kale", "lettuce", "cabbage", "carrot", "zucchini", "cucumber", "asparagus", "cauliflower",
    "tomato", "bell pepper", "green beans", "sweet potato", "pumpkin", "avocado",
    "oatmeal", "brown rice", "quinoa", "whole wheat bread", "whole wheat pasta",
    "chicken breast", "salmon", "tuna", "tofu", "eggs", "lentils", "black beans", "chickpeas",
    "yogurt", "greek yogurt", "skim milk", "soy milk", "almond milk",
    "nuts", "walnuts", "almonds", "cashews", "chia seeds", "flax seeds",
    "olive oil", "hummus", "miso", "soup (vegetable)", "green smoothie", "fruit salad"]
    unhealthy_foods = ["pizza", "cheeseburger", "hamburger", "hotdog", "fried chicken", "french fries", "chips", "nachos",
    "donut", "cake", "cupcake", "cookies", "ice cream", "chocolate bar", "candy", "gummy bears", "toffee",
    "soda", "soft drink", "cola", "energy drink", "milkshake",
    "white bread", "white rice", "sugary cereal", "ice_cream" "frosted flakes", "pop tarts",
    "bacon", "sausage", "processed meat", "salami",
    "microwave meals", "deep-fried snacks",
    "butter popcorn", "cheese puffs", "fast food", "takeaway box"]
    neutral_foods = ["instant noodles", "ramen noodles", "white rice", "white bread", "bagel", "pasta", "noodles", "mac and cheese", "croissant", "pancake", "waffle",
    "mashed potato", "hash browns", "gravy", "mayonnaise", "ketchup", "cheese", "cream cheese",
    "peanut butter", "jam", "honey", "granola", "muffin", "trail mix",
    "beef", "pork", "meatballs", "lamb", "steak", "rotisserie chicken",
    "whole milk", "ice latte", "flavored yogurt", "fruit juice", "sports drink",
    "smoothie (with added sugar)", "suger smoothie", "instant oats",
    "canned soup", "baked beans", "rice cakes", "pretzel"]
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        top_scores, top_indices = torch.topk(probabilities, 5)

    best_label = None
    best_score = 0.0

    for score, idx in zip(top_scores, top_indices):
        label = labels[idx]
        if label.lower() in healthy_foods or label.lower() in unhealthy_foods or label.lower() in neutral_foods:
            best_label = label
            best_score = score.item()
            break
    st.markdown("""
        <style>
         .stApp {
        background: linear-gradient(135deg, #d8f3dc, #95d5b2, #52b788);
        background-size: 400% 400%;
        animation: gradientMove 15s ease infinite;
        color: #1b4332;
    }

    @keyframes gradientMove {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
        </style>
    """, unsafe_allow_html=True)
    if best_label:
        st.markdown(f"**I think it's:** `{best_label.title()}` ({best_score * 100:.2f}% sure)")
        food_name = best_label.lower()
    else:
        st.subheader("** I don't know about this food yet😅. Can you tell me what it is?**")
        food_name = st.text_input("Type the name of the food", placeholder="e.g. chocolate cake, salad, apple...")
        if not food_name:
            st.stop()

    # Health message based on food
    food_name = food_name.lower()

    if any(f in food_name for f in healthy_foods):
        message = "Great job! That’s super healthy!"
        emoji = emojis['healthy']
    elif any(f in food_name for f in unhealthy_foods):
        message = "Hmm... Try to balance that with something healthy!"
        emoji = emojis['unhealthy']
    elif any(f in food_name for f in neutral_foods):
        message = "It is OK to eat in moderation, but don’t overdo it."
        emoji = emojis['neutral']
    else:
        message = "I'm not sure..."
        emoji = emojis['unidentified']

    # Show feedback
    st.markdown(f"### {message} {emoji}")

    # Tip/Fun Facts + try again

    st.subheader("💡 Tips and Fun Facts")
    st.info(random.choice(tips_and_fun_facts))

    if st.button("🔁 Try another food"):
        st.rerun()

# Custom background
st.markdown("""
    <style>
    /* Enhanced Camera Button */
    .take-photo-button button {
        background: linear-gradient(45deg, #ff9a9e, #fad0c4);
        color: #4a148c;
        font-size: 1.3rem;
        font-weight: bold;
        padding: 1rem 2rem;
        border-radius: 1.5rem;
        border: none;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
        animation: pulse 2s infinite;
        position: relative;
        overflow: hidden;
    }

    .take-photo-button button:hover {
        background: linear-gradient(45deg, #fbc2eb, #a6c1ee);
        transform: scale(1.1);
        cursor: pointer;
    }

/* Sparkles on button */
    .take-photo-button button::after {
        content: "✨";
        font-size: 1.5rem;
        position: absolute;
        right: 15px;
        top: 10px;
        animation: twinkle 1s infinite alternate;
    }

/* Camera bounce animation */
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.04); }
        100% { transform: scale(1); }
    }

@keyframes twinkle {
    0% { transform: rotate(0deg) scale(1); opacity: 1; }
    100% { transform: rotate(10deg) scale(1.2); opacity: 0.6; }
}

/* Optional: hide default camera input label */
    .stCameraInput label {
        display: none;
    }
    
    
    
    /* Gradient background */
    .stApp {
        background: linear-gradient(135deg, #d8f3dc, #95d5b2, #52b788);
        background-size: 400% 400%;
        animation: gradientMove 15s ease infinite;
        color: #1b4332;
    }

    @keyframes gradientMove {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

# Fun buttons

    /* Button style */
    .stButton > button {
        background-color: #52b788;
        color: white;
        font-weight: bold;
        border-radius: 12px;
        border: none;
        padding: 0.5em 1.2em;
        transition: 0.3s;
    }

    .stButton > button:hover {
        background-color: #40916c;
        transform: scale(1.05);
        cursor: pointer;
    }

    /* Uploader hover */
    .stFileUploader:hover {
        border: 2px dashed #1b4332;
    }
    
    </style>
""", unsafe_allow_html=True)
import streamlit.components.v1 as components

components.html(
    """
    <audio id="snapSound" src="https://www.soundjay.com/mechanical/camera-shutter-click-02.mp3"></audio>
    <script>
    const snapBtn = window.parent.document.querySelector('button[kind="primary"]');
    if (snapBtn) {
        snapBtn.addEventListener("click", () => {
            const sound = window.parent.document.getElementById("snapSound");
            if (sound) {
                sound.play();
            }
        });
    }
    </script>
    """,
    height=0
)
