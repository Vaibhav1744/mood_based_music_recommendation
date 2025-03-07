# ========== STREAMLIT CONFIG (MUST BE FIRST) ========== #
import streamlit as st
st.set_page_config(
    page_title="Emotion Music Recommender",
    page_icon="🎵",
    layout="centered"
)

# ========== IMPORTS ========== #
import numpy as np
import cv2
import pandas as pd
import os
from collections import Counter
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

# ========== ENVIRONMENT SETUP ========== #
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'

# ========== CONSTANTS ========== #
MODEL_PATH = "model.h5"
DATA_PATH = "muse_v3.csv"
HAARCASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

# ========== MODEL ARCHITECTURE ========== #
def create_emotion_model():
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D((2,2)),
        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D((2,2)),
        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(1024, activation='relu'),
        Dense(7, activation='softmax')
    ])
    return model

# ========== INITIALIZATION ========== #
try:
    model = create_emotion_model()
    model.load_weights(MODEL_PATH)
except Exception as e:
    st.error(f"🚨 Model Loading Error: {str(e)}")
    st.stop()

# ========== DATA LOADING ========== #
@st.cache_data
def load_music_data():
    try:
        df = pd.read_csv(DATA_PATH)
        return df.rename(columns={
            'track': 'name',
            'lastfm_url': 'link',
            'number_of_emotion_tags': 'emotional',
            'valence_tags': 'pleasant'
        })[['name', 'emotional', 'pleasant', 'link', 'artist']]
    except Exception as e:
        st.error(f"📁 Data Loading Error: {str(e)}")
        st.stop()

music_df = load_music_data().sort_values(by=["emotional", "pleasant"])

# ========== MUSIC RECOMMENDATION ENGINE ========== #
def get_recommendations(emotions):
    emotion_pools = {
        'Angry': music_df[36000:54000],
        'Fearful': music_df[18000:36000],
        'Happy': music_df[72000:],
        'Neutral': music_df[54000:72000],
        'Sad': music_df[:18000]
    }
    
    distribution = {
        1: 30,    # Single emotion: 30 songs
        2: [18, 12],  # Two emotions
        3: [15, 10, 5],  # Three emotions
        4: [12, 9, 6, 3],  # Four emotions
        5: [10, 8, 6, 4, 2]  # Five emotions
    }
    
    samples = []
    counts = distribution.get(len(emotions), [10]*len(emotions))
    
    for emotion, count in zip(emotions, counts):
        pool = emotion_pools.get(emotion, emotion_pools['Sad'])
        samples.append(pool.sample(min(count, len(pool))))
    
    return pd.concat(samples).drop_duplicates().head(30)

# ========== STREAMLIT UI ========== #
st.markdown("""
    <style>
    .main {background: #f8f9fa}
    .stButton>button {
        border-radius: 25px;
        padding: 12px 40px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    </style>
    <h1 style='text-align: center; color: #2c3e50; margin-bottom: 30px;'>
        🎧 Emotion-Powered Music Discovery
    </h1>
""", unsafe_allow_html=True)

# ========== EMOTION DETECTION ========== #
if 'detected_emotions' not in st.session_state:
    st.session_state.detected_emotions = []

if st.button("📸 Start Real-Time Emotion Scan", type="primary", use_container_width=True):
    cap = cv2.VideoCapture(0)
    image_placeholder = st.empty()
    progress_bar = st.progress(0)
    
    try:
        for frame_count in range(75):  # Process 75 frames (~5 seconds)
            ret, frame = cap.read()
            if not ret: continue
            
            # Convert frame to RGB for Streamlit display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Enhanced face detection
            faces = cv2.CascadeClassifier(HAARCASCADE_PATH).detectMultiScale(
                gray,
                scaleFactor=1.08,
                minNeighbors=8,
                minSize=(150, 150),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            for (x, y, w, h) in faces:
                try:
                    # Process face region
                    face_roi = cv2.resize(gray[y:y+h, x:x+w], (48, 48))
                    face_roi = np.expand_dims(face_roi / 255.0, axis=(0, -1))
                    
                    # Get prediction with confidence
                    pred = model.predict(face_roi, verbose=0)[0]
                    emotion_idx = np.argmax(pred)
                    confidence = pred[emotion_idx]
                    
                    if confidence > 0.7:  # Only consider confident predictions
                        emotion = ["Angry", "Disgusted", "Fearful",
                                  "Happy", "Neutral", "Sad", "Surprised"][emotion_idx]
                        st.session_state.detected_emotions.append(emotion)
                        
                        # Draw annotations directly on RGB frame
                        cv2.rectangle(frame_rgb, (x, y), (x+w, y+h), (0, 230, 0), 3)
                        cv2.putText(frame_rgb, f"{emotion} {confidence:.0%}",
                                  (x+10, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                  (0, 230, 0), 2)
                
                except Exception as e:
                    continue
            
            # Update Streamlit display
            image_placeholder.image(frame_rgb, caption="Live Emotion Analysis")
            progress_bar.progress((frame_count + 1) / 75)
            
    finally:
        cap.release()

# ========== RESULTS DISPLAY ========== #
if st.session_state.detected_emotions:
    emotion_counts = Counter(st.session_state.detected_emotions)
    if emotion_counts:
        main_emotions = [emotion for emotion, _ in emotion_counts.most_common(3)]
        st.success(f"🎯 Detected Emotions: {', '.join(main_emotions)}")
        
        recommendations = get_recommendations(main_emotions)
        
        st.markdown("## 🎶 Your Personalized Recommendations")
        cols = st.columns(3)
        
        for idx, (_, row) in enumerate(recommendations.iterrows()):
            with cols[idx % 3]:
                st.markdown(f"""
                    <div style="padding:15px; margin:10px 0; background:#ffffff;
                                border-radius:12px; box-shadow:0 4px 12px rgba(0,0,0,0.1);
                                transition: all 0.3s ease;">
                        <a href="{row['link']}" target="_blank" 
                           style="text-decoration:none; color:#2c3e50;">
                            <h4 style="margin:0; font-size:1.1rem;">{row['name']}</h4>
                            <p style="color:#7f8c8d; margin:8px 0 0 15px; font-size:0.9rem;">
                                🎤 {row['artist']}
                            </p>
                        </a>
                    </div>
                """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ No clear emotions detected. Try again with better lighting!")
else:
    st.markdown("""
        <div style="text-align:center; color:#7f8c8d; margin-top:50px;">
            <h3>Click the scan button to begin your musical journey! 🎶</h3>
        </div>
    """, unsafe_allow_html=True)