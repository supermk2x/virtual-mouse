import cv2
import time
import mediapipe as mp
import pyautogui


pyautogui.FAILSAFE = False


mp_holistic = mp.solutions.holistic
holistic_model = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


mp_drawing = mp.solutions.drawing_utils

# Initialize the mouse control parameters
mouse_speed = 20  # Adjust mouse movement speed
click_threshold = 100  # Threshold for mouse click detection


prev_hand_x, prev_hand_y = 0, 0

# OpenCV Video Capture
capture = cv2.VideoCapture(0)

while capture.isOpened():
    ret, frame = capture.read()

    
    frame = cv2.resize(frame, (800, 600))

    
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    
    image.flags.writeable = False
    results = holistic_model.process(image)
    image.flags.writeable = True

    
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    mp_drawing.draw_landmarks(
	image,
	results.right_hand_landmarks,
	mp_holistic.HAND_CONNECTIONS
	)
    # Get hand landmarks
    if results.right_hand_landmarks is not None:
        hand_landmarks = results.right_hand_landmarks.landmark

        # Extract the coordinates of the index and middle fingers
        index_finger_x = int(hand_landmarks[mp_holistic.HandLandmark.INDEX_FINGER_TIP].x * frame.shape[1])
        index_finger_y = int(hand_landmarks[mp_holistic.HandLandmark.INDEX_FINGER_TIP].y * frame.shape[0])
        middle_finger_x = int(hand_landmarks[mp_holistic.HandLandmark.MIDDLE_FINGER_TIP].x * frame.shape[1])
        middle_finger_y = int(hand_landmarks[mp_holistic.HandLandmark.MIDDLE_FINGER_TIP].y * frame.shape[0])

        # Calculate the hand's movement direction
        hand_dx = index_finger_x - prev_hand_x
        hand_dy = index_finger_y - prev_hand_y

        # Move the mouse cursor
        pyautogui.moveRel(hand_dx * mouse_speed, hand_dy * mouse_speed)

        # Update previous hand position
        prev_hand_x, prev_hand_y = index_finger_x, index_finger_y

        
        finger_distance = ((index_finger_x - middle_finger_x) ** 2 + (index_finger_y - middle_finger_y) ** 2) ** 0.5

        
        if finger_distance < click_threshold:
            pyautogui.click(button='left')
        else:
            pyautogui.click(button='right')
    
    cv2.imshow("AI Virtual Mouse", image)

    # Exit the application
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close OpenCV windows
capture.release()
cv2.destroyAllWindows()
