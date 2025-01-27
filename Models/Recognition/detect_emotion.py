from moviepy import *
from fer import FER
import cv2


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


def main():
    """
    Capture la vidéo en direct, détecte les visages et reconnaît les émotions.
    """
    # Initialiser le détecteur
    detector = initialize_detector(mtcnn=True)

    # Capture vidéo
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Reconnaissance des émotions
        frame_with_emotions = detect_emotions(frame, detector)

        # Afficher le flux vidéo
        cv2.imshow("Reconnaissance des émotions", frame_with_emotions)

        # Quitter avec 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
