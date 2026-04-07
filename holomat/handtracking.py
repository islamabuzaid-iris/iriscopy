import cv2
import mediapipe as mp 

#intialize mediapipe hand solution
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.1,
                       min_tracking_confidence=0.1)

mp_drawing = mp.solutions.drawing_utils

#accessing camera
cap = cv2.VideoCapture(0)

#error check
if not cap.isOpened():
    print("Error camera not open")
    exit()



#main loop
while True:
    #capture fbf from camera
    success, frame = cap.read()
    if not success:
        break
    

    #flip camera 
    frame = cv2.flip(frame, 1)

    #convert bgr to rgb\
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    #process frame with mediapipe hand solution
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
           #drawing landmaks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    #draw hand annotations
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            #drawing landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    #display the frame
    cv2.imshow("frame", frame)
    cv2.waitKey(1)

#release capture
cap.release()
cv2.destroyAllWindows()
