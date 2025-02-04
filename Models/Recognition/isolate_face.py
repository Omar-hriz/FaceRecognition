import cv2
import streamlit as st
from fer import FER
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Initialiser le détecteur FER
def initialize_detector(mtcnn=False):
    """Initialise le détecteur FER."""
    return FER(mtcnn=mtcnn)


def detect_emotions(frame, detector):
    """
    Détecte les visages et reconnaît les émotions sur une image.
    """
    results = detector.detect_emotions(frame)
    emotion_details = []

    for result in results:
        box = result['box']
        emotions = result['emotions']
        emotion_details.append(emotions)

        x, y, width, height = box
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
        top_emotion = max(emotions, key=emotions.get)
        text = f"{top_emotion} ({emotions[top_emotion]:.2f})"
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return frame, emotion_details

# Initialiser le détecteur FER
detector = initialize_detector(mtcnn=True)

# Initialiser la caméra
cap = cv2.VideoCapture(0)

# Créer l'interface Streamlit
st.set_page_config(layout="wide")
st.title("Reconnaissance des émotions en temps réel")

# Barre latérale
with st.sidebar:
    st.markdown(
        """
        <div style="text-align: center;">
            <h2>Master 2 IA - 04/02/2025</h2>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.image("logo.png", width=150)
    
    st.markdown(
        """
        <div style="text-align: center;">
            <h2>Projet fil rouge</h2>
            <p><strong>Mathieu Suchet - Omar Hriz - Matthieu Pelissier</strong></p>
        </div>
        """,
        unsafe_allow_html=True
    )

# Centrer la caméra sur la page principale
col_center = st.columns([1, 3, 1])[1]  # Seule la colonne du milieu affiche le flux vidéo

with col_center:
    frame_placeholder = st.empty()
    emotion_placeholder = st.empty()
    chart_placeholder = st.empty()

emotion_history = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_with_emotions, emotion_details = detect_emotions(frame, detector)
    frame_rgb = cv2.cvtColor(frame_with_emotions, cv2.COLOR_BGR2RGB)

    frame_placeholder.image(frame_rgb, channels="RGB", width=700)

    with emotion_placeholder.container():
        st.subheader("Émotions détectées :")
        if emotion_details:
            for i, emotions in enumerate(emotion_details):
                st.markdown(f"### Visage {i + 1}")
                top_emotion = max(emotions, key=emotions.get)
                emotions_list = list(emotions.items())
                emotion_history.append(emotions)
                for j in range(0, len(emotions_list), 3):
                    cols = st.columns(3)
                    for col, (emotion, score) in zip(cols, emotions_list[j:j + 3]):
                        bg_color = "#d4edda" if emotion == top_emotion else "#f4f4f4"
                        text_color = "#155724" if emotion == top_emotion else "#000000"
                        box_shadow = "0px 2px 5px rgba(0, 0, 0, 0.1)" if emotion != top_emotion else "0px 2px 8px rgba(0, 0, 0, 0.3)"
                        border_style = "solid 2px black"
                        with col:
                            st.markdown(
                                f"""
                                <div style="background-color: {bg_color}; padding: 10px; border-radius: 10px; text-align: center; 
                                            box-shadow: {box_shadow}; color: {text_color}; border: {border_style};">
                                    <h4 style="margin: 0;">{emotion}</h4>
                                    <p style="margin: 0; font-size: 18px; font-weight: bold;">{score:.2%}</p>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
        else:
            st.write("Aucune émotion détectée.")
    
    # Affichage de l'évolution des émotions
    if emotion_history:
        df = pd.DataFrame(emotion_history).rolling(5).mean()
        fig, ax = plt.subplots()
        df.plot(ax=ax)
        ax.set_title("Évolution des émotions au fil du temps")
        ax.set_xlabel("Temps")
        ax.set_ylabel("Score des émotions")
        chart_placeholder.pyplot(fig)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
