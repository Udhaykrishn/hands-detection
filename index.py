import cv2 as cv
import mediapipe.python.solutions.hands as Hands
import mediapipe.python.solutions.drawing_utils as draw

hands = Hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
)

cam = cv.VideoCapture(0)

while True:
    success, frame = cam.read()
    
    if not success:
        print("Camera not detected")
        
    frame = cv.flip(frame, 1)
    
    frameRGB = cv.cvtColor(frame,cv.COLOR_RGB2BGR)
    
    handsDelected = hands.process(frameRGB)
    
    
    
    if handsDelected.multi_hand_landmarks:
        for land_mark in handsDelected.multi_hand_landmarks:
            
            draw.draw_landmarks(
                image=frame,
                landmark_list=land_mark,
                connections=Hands.HAND_CONNECTIONS
            )
            
    cv.imshow("Hand Landmark", frame)
    
    if cv.waitKey(1) == ord('q'):
        break
cam.release()
cv.destroyAllWindows()