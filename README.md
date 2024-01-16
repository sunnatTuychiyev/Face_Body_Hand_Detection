# Face-Body-Hand-Pose-Detection

This program utilizes face, body, hand, and pose detectors using the mediapipe library to analyze images and video streams.


## Installation

To install this program, follow these steps:

1. Install the required libraries `cv2` and `mediapipe`:
   ```bash
   pip install opencv-python mediapipe
   
  Clone the repository containing the holistic_detection.py file:
  
      git clone <reponing_git_manziili>
  Run the holistic_detection.py file:
  
      python holistic_detection.py
      
<h2>Additional Information</h2>
  Press the q key to control the program.

<h2>Precautions</h2>
  Make sure to install the required libraries before running the program.

  Face Detector: Outlines the center of detected faces in red, using results.face_landmarks.
  Pose Detector: Identifies the body's landmarks and draws connections using results.pose_landmarks.
  Hand Detectors: Identifies and displays landmarks for the right and left hands using results.right_hand_landmarks     and results.left_hand_landmarks.
  Upon execution, the program will display the landmarks for face, body, hands, and connections on the screen.

<h2>Tips</h2>
  Ensure your camera is enabled to observe the program in action.

<h2>Contact</h2>
  For any questions or suggestions, feel free to contact me via email: uzchit77@gmail.com.
