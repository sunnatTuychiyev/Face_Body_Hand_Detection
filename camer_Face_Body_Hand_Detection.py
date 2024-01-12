import cv2
import mediapipe as mp

# Initialize MediaPipe holistic model for face, body, hand, and pose detection
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.2, min_tracking_confidence=0.2)

# Open camera stream
cap = cv2.VideoCapture(0)

# Skeleton position
sklet_x, sklet_y = 50, 50

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Process face, body, hand, and pose detection using MediaPipe
    results = holistic.process(frame)

    # Add a black background
    frame[:] = [0, 0, 0]

    # Uncomment the following line to display the skeleton
    # cv2.putText(frame, 'Skeleton', (sklet_x, sklet_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Draw detected faces
    if results.face_landmarks:
        for landmark in results.face_landmarks.landmark:
            ih, iw, _ = frame.shape
            x, y = int(landmark.x * iw), int(landmark.y * ih)
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

    # Draw detected body landmarks
    if results.pose_landmarks:
        for landmark in results.pose_landmarks.landmark:
            ih, iw, _ = frame.shape
            x, y = int(landmark.x * iw), int(landmark.y * ih)
            cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

    # Draw detected hand landmarks
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

    # Draw body connections
    if results.pose_landmarks:
        # Connect the anatomical landmarks
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        # Draw connections for hands and body
        mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    # Display on the screen
    cv2.imshow('Holistic Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
