import cv2
# Charger le classificateur
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

image_size = (128, 128, 3)

# Initialiser la caméra
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()  # Lire une image de la caméra

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Détecter les visages
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    # Dessiner des rectangles autour des visages

    if len(faces) <= 2:
        for (x, y, w, h) in faces:
            # cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            # frame[y:y+h, x:x+w] = cv2.GaussianBlur(frame[y:y+h, x:x+w], (51, 51), 20)
            cv2.imshow('Visages détectés', frame)
            
            
            isolated_face = frame[y:y+h, x:x+w]
            isolated_face = cv2.resize(isolated_face, image_size[:2])
            
            # Model guess
            # prediction = model.predict(isolated_face)
            
            

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break