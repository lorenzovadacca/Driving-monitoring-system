# Import the needed libraries
import cv2
import mediapipe as mp
import numpy as np 
import time

# Set the desired setting
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces = 1,
    refine_landmarks = True, # Enables  detailed eyes points
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5
)
mp_drawing_styles = mp.solutions.drawing_styles
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Open the video source
cap = cv2.VideoCapture(0) # Local webcam (index start from 0)

# Define variables outside the loop
high_ear_start_time = None
t1 = t2 = t3 = t4 = 0
t1_start = t2_start = t3_start = t4_start = 0
in_closing = False
in_closed = False
in_opening = False
in_opened = False

# Iterate (within an infinite loop)
while cap.isOpened(): 
    
    # Get the new frame
    success, image = cap.read()

    start = time.time()

    # Also convert the color space from BGR to RGB
    if image is None:
        break

    # To improve performace
    image.flags.writeable = False
    
    # Run MediaPipe on the frame
    results = face_mesh.process(image)

    # To improve performance
    image.flags.writeable = True

    img_h, img_w, img_c = image.shape

    # Define variables inside the loop
    point_RER = [] # Right Eye Right
    point_REB = [] # Right Eye Bottom
    point_REL = [] # Right Eye Left
    point_RET = [] # Right Eye Top

    point_LER = [] # Left Eye Right
    point_LEB = [] # Left Eye Bottom
    point_LEL = [] # Left Eye Left
    point_LET = [] # Left Eye Top

    point_REIC = [] # Right Eye Iris Center
    point_LEIC = [] # Left Eye Iris Center

    face_2d = [] #face2d
    face_3d = [] #face3d

    right_eye_2d = [] #right_eye_2d
    left_eye_2d = [] #left_eye_2d

    right_eye_3d = [] #right_eye_3d
    left_eye_3d = [] #left_eye_3d

    # Get the landmark coordinates
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):

                # Eye Gaze (Iris Tracking)
                # Left eye indices list
                #LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
                # Right eye indices list
                #RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]
                #LEFT_IRIS = [473, 474, 475, 476, 477]
                #RIGHT_IRIS = [468, 469, 470, 471, 472]

                if idx == 33:
                    point_RER = (lm.x * img_w, lm.y * img_h)
                    cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=5, color=(0, 0, 255), thickness=-1)
                if idx == 145:
                    point_REB = (lm.x * img_w, lm.y * img_h)
                    cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=5, color=(0, 0, 255), thickness=-1)
                if idx == 133:
                    point_REL = (lm.x * img_w, lm.y * img_h)
                    cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=5, color=(0, 0, 255), thickness=-1)
                if idx == 159:
                    point_RET = (lm.x * img_w, lm.y * img_h)
                    cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=5, color=(0, 0, 255), thickness=-1)
                if idx == 362:
                    point_LER = (lm.x * img_w, lm.y * img_h)
                if idx == 374:
                    point_LEB = (lm.x * img_w, lm.y * img_h)
                if idx == 263:
                    point_LEL = (lm.x * img_w, lm.y * img_h)
                if idx == 386:
                    point_LET = (lm.x * img_w, lm.y * img_h)
                if idx == 468:
                    point_REIC = (lm.x * img_w, lm.y * img_h)
                if idx == 469:
                    point_469 = (lm.x * img_w, lm.y * img_h)
                if idx == 470:
                    point_470 = (lm.x * img_w, lm.y * img_h)
                if idx == 471:
                    point_471 = (lm.x * img_w, lm.y * img_h)
                if idx == 472:
                    point_472 = (lm.x * img_w, lm.y * img_h)
                if idx == 473:
                    point_LEIC = (lm.x * img_w, lm.y * img_h)
                if idx == 474:
                    point_474 = (lm.x * img_w, lm.y * img_h)
                if idx == 475:
                    point_475 = (lm.x * img_w, lm.y * img_h)
                if idx == 476:
                    point_476 = (lm.x * img_w, lm.y * img_h)
                if idx == 477:
                    point_477 = (lm.x * img_w, lm.y * img_h)
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
                    x, y = int(lm.x * img_w), int(lm.y * img_h)

                    # Get the 2D Coordinates
                    face_2d.append([x,y])

                    # Get the 3D Coordinates
                    face_3d.append([x,y, lm.z])

                #LEFT_IRIS = [473, 474, 475, 476, 477]
                if idx == 473 or idx == 362 or idx == 374 or idx == 263 or idx == 386: # iris points
                    if idx == 473:
                        left_pupil_2d = (lm.x * img_w, lm.y * img_h)
                        left_pupil_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
                    x, y = int(lm.x * img_w), int(lm.y * img_h)

                    left_eye_2d.append([x,y])
                    left_eye_3d.append([x,y,lm.z])

                #RIGHT_IRIS = [468, 469, 470, 471, 472]
                if idx == 468 or idx == 33 or idx == 145 or idx == 133 or idx == 159: # iris points
                    if idx == 468:
                        right_pupil_2d = (lm.x * img_w, lm.y * img_h)
                        right_pupil_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
                    x, y = int(lm.x * img_w), int(lm.y * img_h)

                    right_eye_2d.append([x,y])
                    right_eye_3d.append([x,y,lm.z])
        
        face_2d = np.array(face_2d,dtype=np.float64)
        face_3d = np.array(face_3d,dtype=np.float64)
        left_eye_2d = np.array(left_eye_2d,dtype=np.float64)
        left_eye_3d = np.array(left_eye_3d,dtype=np.float64)
        right_eye_2d = np.array(right_eye_2d,dtype=np.float64)
        right_eye_3d = np.array(right_eye_3d,dtype=np.float64)
        
        # The camera matrix
        focal_lenght= 1 * img_w

        cam_matrix = np.array([ [focal_lenght, 0, img_h/2], [0, focal_lenght, img_w/2], [0,0,1]])
        
        # The distorsion parameters
        dist_matrix = np.zeros((4,1), dtype=np.float64)
        
        # Solve PnP
        success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
        succes_left_eye, rot_vec_left_eye, trans_vec_left_eye = cv2.solvePnP(left_eye_3d, left_eye_2d, cam_matrix, dist_matrix)
        success_right_eye, rot_vec_right_eye, trans_vec_right_eye = cv2.solvePnP(right_eye_3d, right_eye_2d, cam_matrix, dist_matrix)

        # Get rotational matrix
        rmat, jac = cv2.Rodrigues(rot_vec)
        rmat_left_eye, jac_left_eye = cv2.Rodrigues(rot_vec_left_eye)
        rmat_right_eye, jac_right_eye = cv2.Rodrigues(rot_vec_right_eye)

        # Get angles
        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
        angles_left_eye, mtrx_left_eye, mtxQ_left_eye, Qx_left_eye, Qy_left_eye, Qz_left_eye = cv2.RQDecomp3x3(rmat_left_eye)
        angles_right_eye, mtxR_right_eye, mtxQ_right_eye, Qx_right_eye, Qy_right_eye, Qz_right_eye = cv2.RQDecomp3x3(rmat_right_eye)

        # Convert angles in degrees
        pitch = angles[0] * 1800
        yaw = -angles[1] * 1800
        roll = 180 + (np.arctan2(point_RER[1] - point_LEL[1], point_RER[0]- point_LEL[0]) * 180 / np.pi)
        if roll > 180:
            roll = roll-360

        pitch_left_eye = angles_left_eye[0] * 1800
        yaw_left_eye =angles_left_eye[1] * 1800

        pitch_right_eye = angles_right_eye[0] * 1800
        yaw_right_eye = angles_right_eye[1] * 1800

        # Display directions
        nose_3d_projection, jacobian = cv2.projectPoints(nose_3d , rot_vec , trans_vec , cam_matrix , dist_matrix)
        p1_nose = (int(nose_2d[0]), int(nose_2d[1]))
        p2_nose = (int(nose_2d[0] - yaw * 10), int(nose_2d[1] - pitch * 10))
        cv2.line(image, p1_nose, p2_nose, (255,0,0), 3)

        left_eye_3d_projection, jacobian = cv2.projectPoints(left_eye_3d , rot_vec , trans_vec , cam_matrix , dist_matrix)
        x_left_eye_2d, y_left_eye_2d = left_eye_2d[4]
        p1_left_eye = (int(x_left_eye_2d), int(y_left_eye_2d))
        p2_left_eye = (int(x_left_eye_2d + yaw * 10), int(y_left_eye_2d - pitch * 10))
        cv2.line(image, p1_left_eye, p2_left_eye, (255,0,0), 3)

        right_eye_3d_projection, jacobian = cv2.projectPoints(right_eye_3d , rot_vec , trans_vec , cam_matrix , dist_matrix)
        x_right_eye_2d, y_right_eye_2d = right_eye_2d[4]
        p1_right_eye = (int(x_right_eye_2d), int(y_right_eye_2d))
        p2_right_eye = (int(x_right_eye_2d + yaw * 10), int(y_right_eye_2d - pitch * 10))
        cv2.line(image, p1_right_eye, p2_right_eye, (255,0,0), 3)
            
        # Draw the positions on the frame
        l_eye_width = point_LEL[0] - point_LER[0]
        l_eye_height = point_LEB[1] - point_LET[1]
        l_eye_center = [(point_LEL[0] + point_LER[0])/2 ,(point_LEB[1] + point_LET[1])/2]
        cv2.circle(image, (int(point_LEIC[0]), int(point_LEIC[1])), radius=3, color=(0, 255, 0), thickness=-1) # Center of iris
        cv2.circle(image, (int(l_eye_center[0]), int(l_eye_center[1])), radius=2, color=(128, 128, 128), thickness=-1) # Center of eye
        cv2.putText(image, "Left eye:  x = " + str(np.round(point_LEIC[0],0)) + " , y = " + str(np.round(point_LEIC[1],0)), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2) 

        r_eye_width = point_REL[0] - point_RER[0]
        r_eye_height = point_REB[1] - point_RET[1]
        r_eye_center = [(point_REL[0] + point_RER[0])/2 ,(point_REB[1] + point_RET[1])/2]
        cv2.circle(image, (int(point_REIC[0]), int(point_REIC[1])), radius=3, color=(0, 0, 255), thickness=-1) # Center of iris
        cv2.circle(image, (int(r_eye_center[0]), int(r_eye_center[1])), radius=2, color=(128, 128, 128), thickness=-1) # Center of eye
        cv2.putText(image, "Right eye: x = " + str(np.round(point_REIC[0],0)) + " , y = " + str(np.round(point_REIC[1],0)), (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2) 
            
        # Speed reduction
        time.sleep(1/25) # [s]

        end = time.time()
        totalTime = end-start

        if totalTime>0:
            fps = 1 / totalTime
        else:
            fps=0

    # Defining variable and thresholds
    HIGH_EAR = 0.8
    PERCLOS_ALERT_THRESHOLD = 0.4   # supposed to be 0.4
    DISTRACTION_THRESHOLD = 30
    HIGH_EAR_DURATION = 10  # seconds
    current_time = time.time()

    # Gaze deviation alert
    if 'pitch' in locals() and 'yaw' in locals() and 'roll' in locals():
        combined_deviation = max(abs(pitch), abs(yaw), abs(roll))
        if combined_deviation > DISTRACTION_THRESHOLD:
            cv2.putText(image, "ALERT: Driver Distracted (Gaze Deviation)", (20, 250), cv2.FONT_HERSHEY_SIMPLEX, 1 , (0, 0, 255), 2)   

    # Define EAR calculation function

    def calculate_ear(eye_top, eye_bottom, eye_left, eye_right):
        vertical_1 = abs(eye_top[1] - eye_bottom[1])
        horizontal = abs(eye_left[0] - eye_right[0])
        return vertical_1 / horizontal if horizontal != 0 else 0


    # Compute EAR
    if point_LEL and point_LER and point_LET and point_LEB:
        left_ear = calculate_ear(point_LET, point_LEB, point_LEL, point_LER)
    else:
        left_ear = 0

    if point_REL and point_RER and point_RET and point_REB:
        right_ear = calculate_ear(point_RET, point_REB, point_REL, point_RER)
    else:
        right_ear = 0

    # Get average EAR between the two eyes
    avg_ear = (left_ear + right_ear) / 2

    # Normalize EAR between 0 and 1 in order to match threshold requirement
    normalized_ear = 3.333 * avg_ear - 0.5
    normalized_ear = max(0.0, min(1.0, normalized_ear))  # clamp between 0 and 1

    # Determine EAR state
    if normalized_ear > HIGH_EAR:
        if high_ear_start_time is None:
            high_ear_start_time = current_time
        elif current_time - high_ear_start_time >= HIGH_EAR_DURATION: 
            cv2.putText(image, "ALERT: Drowsiness detected (EAR high)", (20, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        high_ear_start_time = None

    
    # Define time instants in order to compute PERCLOS
    if normalized_ear > 0.8 and not in_closing and not in_closed and not in_opening and t1_start == 0:
        t1_start = time.time()
        print("Restart cycle")

    elif not in_opening and not in_closing and 0.2 < normalized_ear <= 0.8 and t1_start != 0:
        t1 = time.time() - t1_start
        t1_start = 0  # Reset for the next cycle
        t2_start = time.time()
        in_closing = True
        print("t1 (100% → 80%):", t1)

    elif in_closing and normalized_ear <= 0.2:
        t2 = time.time() - t2_start
        t3_start = time.time()
        t4_start = time.time()
        in_closing = False
        in_closed = True
        print("t2 (80% → 20%):", t2)

    elif in_closed and normalized_ear >= 0.2:
        t3 = time.time() - t3_start
        in_closed = False
        in_opening = True
        print("t3 (20% → 0 → 20%):", t3)

    elif in_opening and normalized_ear > 0.8:
        t4 = time.time() - t4_start
        in_opening = False
        print("t4 (20% → 0 → 80%):", t4)

    # Calculate PERCLOS
    if (t4 - t1) > 0:
        perclos = (t3 - t2) / (t4 - t1)

        # Reset for next cycle
        t1 = t2 = t3 = t4 = 0
        t1_start = t2_start = t3_start = t4_start = 0
        in_closing = in_closed = in_opening = False

        if perclos > PERCLOS_ALERT_THRESHOLD:
            cv2.putText(image, "ALERT: Drowsiness Detected (PERCLOS)", (20, 350),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
    else:
        perclos = 0
    
    if perclos > 0:
        print("PERCLOS: ", perclos)
    
    # Display EAR and PERCLOS on the image
    cv2.putText(image, f"EAR: {normalized_ear:.2f}", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.putText(image, f"PERCLOS: {perclos:.2f}", (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    # Show the frame to the user
    cv2.imshow('Technologies for Autonomous Vehicles - Driver Monitoring Systems using AI code sample', image)       
                        
    if cv2.waitKey(5) & 0xFF == 27:
        break