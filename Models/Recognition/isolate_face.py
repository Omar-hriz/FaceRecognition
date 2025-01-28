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
        numpy array: Image avec des boîtes englobantes et étiquettes des émotions.
    """
    results = detector.detect_emotions(frame)

    for result in results:
        box = result['box']
        emotions = result['emotions']
        top_emotion, score = detector.top_emotion(frame)

        # Dessiner un rectangle autour du visage
        x, y, width, height = box
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

        # Ajouter le texte de l'émotion dominante
        text = f"{top_emotion} ({score:.2f})"
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return frame


# Initialiser le détecteur FER
detector = initialize_detector(mtcnn=True)

# Initialiser la caméra
cap = cv2.VideoCapture(0)

# Créer l'interface Streamlit
st.title("Reconnaissance des émotions en temps réel")
st.write("Appuyez sur 'q' pour quitter")

# Créer un espace réservé pour l'affichage de la vidéo
frame_placeholder = st.empty()

while True:
    ret, frame = cap.read()  # Lire une image de la caméra
    if not ret:
        break

    # Reconnaître les émotions
    frame_with_emotions = detect_emotions(frame, detector)

    # Convertir l'image pour l'affichage dans Streamlit (en RGB)
    frame_rgb = cv2.cvtColor(frame_with_emotions, cv2.COLOR_BGR2RGB)

    # Afficher la vidéo image par image dans l'espace réservé
    frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

    # Arrêter la capture si l'utilisateur appuie sur 'q' (fonctionne sur Streamlit avec un timeout)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer la caméra et fermer toutes les fenêtres
cap.release()
cv2.destroyAllWindows()
