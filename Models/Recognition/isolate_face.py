import cv2
import streamlit as st
from fer import FER
import numpy as np

# Initialiser le détecteur FER
def initialize_detector(mtcnn=False):
    """Initialise le détecteur FER."""
    return FER(mtcnn=mtcnn)


def detect_emotions(frame, detector):
    """
    Détecte les visages et reconnaît les émotions sur une image.

    Parameters:
        frame (numpy array): Image capturée (frame de la caméra).
        detector (FER): Instance du détecteur FER.

    Returns:
        tuple:
        - numpy array: Image avec des boîtes englobantes et étiquettes des émotions.
        - list of dict: Liste des émotions détectées pour chaque visage.
    """
    results = detector.detect_emotions(frame)
    emotion_details = []  # Pour stocker les émotions et leurs scores

    for result in results:
        box = result['box']
        emotions = result['emotions']  # Dictionnaire des émotions avec leurs scores
        emotion_details.append(emotions)

        # Dessiner un rectangle autour du visage
        x, y, width, height = box
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

        # Afficher l'émotion dominante directement sur l'image
        top_emotion = max(emotions, key=emotions.get)
        text = f"{top_emotion} ({emotions[top_emotion]:.2f})"
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return frame, emotion_details


# Initialiser le détecteur FER
detector = initialize_detector(mtcnn=True)

# Initialiser la caméra
cap = cv2.VideoCapture(0)

# Créer l'interface Streamlit
st.title("Reconnaissance des émotions en temps réel")
st.write("Appuyez sur 'q' pour quitter")

# Créer un espace réservé pour l'affichage de la vidéo
frame_placeholder = st.empty()
emotion_placeholder = st.empty()

while True:
    ret, frame = cap.read()  # Lire une image de la caméra
    if not ret:
        break

    # Reconnaître les émotions
    frame_with_emotions, emotion_details = detect_emotions(frame, detector)

    # Convertir l'image pour l'affichage dans Streamlit (en RGB)
    frame_rgb = cv2.cvtColor(frame_with_emotions, cv2.COLOR_BGR2RGB)

    # Afficher la vidéo image par image dans l'espace réservé
    frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

    # Afficher les émotions détectées sous forme de cartes dans Streamlit
    with emotion_placeholder.container():
        st.subheader("Émotions détectées :")
        if emotion_details:
            for i, emotions in enumerate(emotion_details):
                st.markdown(f"### Visage {i + 1}")
                # Identifier l'émotion avec le pourcentage le plus élevé
                top_emotion = max(emotions, key=emotions.get)

                # Regrouper les cartes par lignes de 3
                emotions_list = list(emotions.items())
                for j in range(0, len(emotions_list), 3):  # Découper en tranches de 3
                    cols = st.columns(3)  # Créer 3 colonnes pour la ligne
                    for col, (emotion, score) in zip(cols, emotions_list[j:j + 3]):
                        # Définir la couleur de la carte (vert pour l'émotion dominante, gris sinon)
                        bg_color = "#d4edda" if emotion == top_emotion else "#f4f4f4"
                        text_color = "#155724" if emotion == top_emotion else "#000000"
                        box_shadow = "0px 2px 5px rgba(0, 0, 0, 0.1)" if emotion != top_emotion else "0px 2px 8px rgba(0, 0, 0, 0.3)"

                        # Ajouter un contour noir à chaque carte
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

    # Arrêter la capture si l'utilisateur appuie sur 'q' (fonctionne sur Streamlit avec un timeout)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer la caméra et fermer toutes les fenêtres
cap.release()
cv2.destroyAllWindows()
