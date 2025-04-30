import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Create a black canvas for drawing
canvas = np.zeros((480, 640, 3), dtype=np.uint8)

# OpenCV video capture
cap = cv2.VideoCapture(0)

# Variables for tracking previous finger position

prev_x, prev_y = None, None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame for a mirrored effect
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame with Mediapipe
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract index finger tip coordinates
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            h, w, _ = frame.shape
            x, y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)

            if prev_x is not None and prev_y is not None:
                # Draw a line from the previous to the current index finger position
                cv2.line(canvas, (prev_x, prev_y), (x, y), (0, 255, 0), 5)

            # Update previous finger position
            prev_x, prev_y = x, y

            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Merge the canvas with the webcam feed
    frame = cv2.addWeighted(frame, 0.7, canvas, 0.3, 0)

    cv2.imshow("AR Drawing", frame)

    # Clear drawing with 'c' key
    if cv2.waitKey(1) & 0xFF == ord('c'):
        canvas = np.zeros((480, 640, 3), dtype=np.uint8)

    # Quit with 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()