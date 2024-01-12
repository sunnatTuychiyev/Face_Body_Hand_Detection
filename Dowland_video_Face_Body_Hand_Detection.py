import cv2
import mediapipe as mp

# Create MediaPipe holistic model for face, pose, hand, and body detection
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.2, min_tracking_confidence=0.2)

# Open the video stream
cap = cv2.VideoCapture('Enter_the_location_of_the_video,mp4')

# Skeleton position
sklet_x, sklet_y = 50, 50

# Prepare video recording
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('Output_Enter_the_location_of_the_video,mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Process face, pose, hand, and body detection using MediaPipe
    results = holistic.process(frame)

    # Add a black background
    frame[:] = [0, 0, 0]

    # Uncomment the following line to display the skeleton
    # cv2.putText(frame, 'Skeleton', (sklet_x, sklet_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display detected faces
    if results.face_landmarks:
        for landmark in results.face_landmarks.landmark:
            ih, iw, _ = frame.shape
            x, y = int(landmark.x * iw), int(landmark.y * ih)
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

    # Display detected pose landmarks
    if results.pose_landmarks:
        for landmark in results.pose_landmarks.landmark:
            ih, iw, _ = frame.shape
            x, y = int(landmark.x * iw), int(landmark.y * ih)
            cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

    # Display detected left hand landmarks
    if results.left_hand_landmarks:
        for landmark in results.left_hand_landmarks.landmark:
            ih, iw, _ = frame.shape
            x, y = int(landmark.x * iw), int(landmark.y * ih)
            cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

    # Display detected right hand landmarks
    if results.right_hand_landmarks:
        for landmark in results.right_hand_landmarks.landmark:
            ih, iw, _ = frame.shape
            x, y = int(landmark.x * iw), int(landmark.y * ih)
            cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

    # Display body landmarks
    if results.pose_landmarks:
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        # Draw connections for hands and body
        mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    # Show the result on the screen
    cv2.imshow('Holistic Detection', frame)

    # Save the video
    out.write(frame)

    # Press 'q' to exit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
