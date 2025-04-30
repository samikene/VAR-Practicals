import cv2
import mediapipe as mp
import numpy as np

# Load the accessory (glasses) image with alpha transparency
glasses_img = cv2.imread("glasses.png", cv2.IMREAD_UNCHANGED)  # Ensure the image has an alpha channel

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Capture video from webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame for a mirrored effect
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame with Face Mesh
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Get left & right eye corner landmarks for scaling
            left_eye = face_landmarks.landmark[33]  # Left eye outer corner
            right_eye = face_landmarks.landmark[263]  # Right eye outer corner

            h, w, _ = frame.shape
            x1, y1 = int(left_eye.x * w), int(left_eye.y * h)
            x2, y2 = int(right_eye.x * w), int(right_eye.y * h)


            # Compute width & height for glasses based on eye distance
            glasses_width = int(1.5 * (x2 - x1))  # Adjust scaling factor
            glasses_height = int(glasses_width * (glasses_img.shape[0] / glasses_img.shape[1]))

            # Positioning the glasses
            x_center = (x1 + x2) // 2
            y_center = (y1 + y2) // 2 + 10  # Adjust vertical alignment
            x1, x2 = x_center - glasses_width // 2, x_center + glasses_width // 2
            y1, y2 = y_center - glasses_height // 2, y_center + glasses_height // 2

            # Resize glasses
            resized_glasses = cv2.resize(glasses_img, (glasses_width, glasses_height))

            # Overlay glasses on face
            if x1 >= 0 and y1 >= 0 and x2 < w and y2 < h:
                for i in range(glasses_height):
                    for j in range(glasses_width):
                        if resized_glasses.shape[2] == 4:  # Check if image has an alpha channel
                            alpha = resized_glasses[i, j, 3] / 255.0
                            if alpha > 0:  # Only overlay non-transparent pixels
                                frame[y1 + i, x1 + j] = (1 - alpha) * frame[y1 + i, x1 + j] + alpha * resized_glasses[i, j, :3]

    cv2.imshow("AR Try-On: Glasses", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
