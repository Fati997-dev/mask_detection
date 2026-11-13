import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Chargement du modèle
model = load_model("best_model.keras")
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

print("Webcam démarrée — appuie sur 'q' pour quitter")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60,60))

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (128, 128))
        face_norm = face_resized.astype("float32") / 255.0
        face_input = np.expand_dims(face_norm, axis=0)

        prob = model.predict(face_input, verbose=0)[0][0]

        if prob >= 0.5:
            label = "Masque OK"
            color = (0, 200, 0)       # vert
            conf  =  prob * 100
        else:
            label = "Sans masque !"
            color = (0, 0, 220)       # rouge
            conf  = (1 - prob) * 100

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.rectangle(frame, (x, y-35), (x+w, y), color, -1)
        cv2.putText(frame, f"{label} {conf:.0f}%",
                    (x+5, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255,255,255), 2)

    cv2.imshow("Detection de masque", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Webcam fermée.")