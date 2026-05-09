from deepface import DeepFace
import cv2

# open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # analyze emotion
    result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

    emotion = result[0]['dominant_emotion']

    # show text on screen
    cv2.putText(frame, emotion, (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # show video
    cv2.imshow("Emotion Detector", frame)

    # press q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()