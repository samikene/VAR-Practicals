import cv2
import numpy as np

# Load the furniture image (make sure the image has a transparent background)
furniture_img = cv2.imread("furniture.png", cv2.IMREAD_UNCHANGED)

# Resize the furniture image (optional)
furniture_img = cv2.resize(furniture_img, (200, 200))  # Adjust size as needed

# Variables to store object position
furniture_position = None

def overlay_transparent(background, overlay, x, y):
    """ Overlays a transparent image on a background """
    h, w, _ = overlay.shape
    bg_h, bg_w, _ = background.shape

    if x >= bg_w or y >= bg_h:
        return background  # If out of bounds, return original frame

    # Ensure the overlay fits in the frame
    w = min(w, bg_w - x)
    h = min(h, bg_h - y)
    overlay = overlay[:h, :w]

    # Extract the alpha channel (transparency)
    alpha = overlay[:, :, 3] / 255.0
    for c in range(3):
        background[y:y+h, x:x+w, c] = (1 - alpha) * background[y:y+h, x:x+w, c] + alpha * overlay[:, :, c]

    return background

def mouse_click(event, x, y, flags, param):
    """ Stores the clicked position to place the furniture """
    global furniture_position
    if event == cv2.EVENT_LBUTTONDOWN:
        furniture_position = (x, y)

# Open webcam
cap = cv2.VideoCapture(0)
cv2.namedWindow("AR Furniture Placement")
cv2.setMouseCallback("AR Furniture Placement", mouse_click)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Mirror the frame for a better experience

    if furniture_position:
        x, y = furniture_position
        frame = overlay_transparent(frame, furniture_img, x - furniture_img.shape[1] // 2, y - furniture_img.shape[0] // 2)

    cv2.putText(frame, "Click on the floor to place furniture", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("AR Furniture Placement", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()
