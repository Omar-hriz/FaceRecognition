import cv2
import streamlit as st
import numpy as np

# Charger le classificateur de visages
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

# Initialiser la caméra
cap = cv2.VideoCapture(0)

# Définir les dimensions de l'image
image_size = (128, 128, 3)

# Créer l'interface Streamlit
st.title("Détection de visages en temps réel")
st.write("Appuyez sur 'q' pour quitter")

# Créer un espace réservé pour l'affichage de la vidéo
frame_placeholder = st.empty()

while True:
    ret, frame = cap.read()  # Lire une image de la caméra
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Détecter les visages
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) <= 2:
        for (x, y, w, h) in faces:
            # Dessiner des rectangles autour des visages
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Isoler le visage détecté
            isolated_face = frame[y:y + h, x:x + w]
            isolated_face = cv2.resize(isolated_face, image_size[:2])

            # Ici vous pouvez ajouter un modèle pour prédire, par exemple
            # prediction = model.predict(isolated_face)

    # Convertir l'image pour l'affichage dans Streamlit (en RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Afficher la vidéo image par image dans l'espace réservé
    frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

    # Arrêter la capture si l'utilisateur appuie sur 'q' (fonctionne sur Streamlit avec un timeout)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer la caméra et fermer toutes les fenêtres
cap.release()
cv2.destroyAllWindows()
