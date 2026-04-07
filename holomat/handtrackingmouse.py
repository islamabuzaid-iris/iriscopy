import cv2
print('cv2 okay')
import mediapipe as mp 
print('mp okay')
import pyautogui as pag
print ('pag okay')
import numpy as np
print ('numpy ok')

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode 
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.1,
    min_tracking_confidence=0.1
)
print('options created')
landmarker=HandLandmarker.create_from_options(options)
print('landmark created')
#accessing camera
cap = cv2.VideoCapture(0)
print("Opened:", cap.isOpened())

#error check
if not cap.isOpened():
    print("Error camera not open")
    exit()

#get screen size
screen_width, screen_height = pag.size()

mouseDown = False

#main loop
while True:
    #capture fbf from camera
    success, frame = cap.read()
    print("Frame:", success)
    if not success:
        break
    

    #flip camera 
    frame = cv2.flip(frame, 1)

    #convert bgr to rgb\
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    #process frame with mediapipe hand solution
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
    results = landmarker.detect_for_video(mp_image, timestamp_ms)


   #frame resolution
    frame_height, frame_width, _ = frame.shape
   
    #draw hand annotations
    if results.hand_landmarks:
        for hand_landmarks in results.hand_landmarks:
            #drawing landmarks
            for lm in hand_landmarks:
                x = int(lm.x * frame_width)
                y = int(lm.y * frame_height)
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

            index_finger_tip = hand_landmarks[8]
            thumb_tip = hand_landmarks[4]

            #get midpoint
            midpoint_x = (index_finger_tip.x + thumb_tip.x) / 2
            midpoint_y = (index_finger_tip.y + thumb_tip.y) / 2

            #get distance
            distance = np.sqrt((index_finger_tip.x - thumb_tip.x) ** 2 + (index_finger_tip.y - thumb_tip.y) ** 2)

            if distance < 0.1 and mouseDown == False:
                #mouse down
                pag.mouseDown()
                mouseDown = True

            if distance > 0.3 and mouseDown == True:
                #mouse up
                pag.mouseUp()
                mouseDown = False
            
            if mouseDown:
                #draw green circle at midpoint
                cv2.circle(frame, (int(midpoint_x * frame_width), int(midpoint_y * frame_height)), 15, (0, 255, 0,),-1)
            else:
                #draw red circle at midpoint
                cv2.circle(frame, (int(midpoint_x * frame_width), int(midpoint_y * frame_height)), 15, (0, 225, 0, 1))

            #map the position to screen resolution
            x_mapped = np.interp(midpoint_x, (0, 1), (0, screen_width))
            y_mapped = np.interp(midpoint_y, (0, 1), (0, screen_height))

            #set mouse
            pag.moveTo(x_mapped, y_mapped, duration=0.025)




    #display the frame
    cv2.imshow("frame", frame)
    cv2.waitKey(1)
    #close program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
#release capture
cap.release()
cv2.destroyAllWindows()