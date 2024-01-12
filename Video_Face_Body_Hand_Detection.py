import cv2
import mediapipe as mp

# MediaPipe tanlashi va yuz, tanasi, qo'l va body detektorlarni yaratish
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.2, min_tracking_confidence=0.2)

# Video fluxini ochish
cap = cv2.VideoCapture('/Users/tuychiyevsunnatillo/Desktop/B.MP4')

# Skletning joylashuvi
sklet_x, sklet_y = 50, 50

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # MediaPipe tanlashi yordamida yuz, tanasi, qo'l va body detektorlarni ishga tushirish
    results = holistic.process(frame)

    # Qora fonni qo'shish
    frame[:] = [0, 0, 0]

    # Skletni chiqarish
  #  cv2.putText(frame, 'Skeleton', (sklet_x, sklet_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Tanilgan yuzlarni chiqarish
    if results.face_landmarks:
        for landmark in results.face_landmarks.landmark:
            ih, iw, _ = frame.shape
            x, y = int(landmark.x * iw), int(landmark.y * ih)
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

    # Tanilgan tanalarni chiqarish
    if results.pose_landmarks:
        for landmark in results.pose_landmarks.landmark:
            ih, iw, _ = frame.shape
            x, y = int(landmark.x * iw), int(landmark.y * ih)
            cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

    # Tanilgan qollarni chiqarish
    if results.left_hand_landmarks:
        for landmark in results.left_hand_landmarks.landmark:
            ih, iw, _ = frame.shape
            x, y = int(landmark.x * iw), int(landmark.y * ih)
            cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

    if results.right_hand_landmarks:
        for landmark in results.right_hand_landmarks.landmark:
            ih, iw, _ = frame.shape
            x, y = int(landmark.x * iw), int(landmark.y * ih)
            cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

    # Body detektori
    if results.pose_landmarks:
        # Anatomik nuqtalarni bog'lab chizish
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        # Qo'l va body chizish uchun chiziqlar
        mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    # Ekranga chiqarish
    cv2.imshow('Holistic Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Yopish
cap.release()
cv2.destroyAllWindows()
