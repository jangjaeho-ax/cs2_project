import cv2
import mediapipe as mp
import numpy as np
from gtts import gTTS
from playsound import playsound
import tensorflow as tf
from tensorflow.keras.models import load_model
def main():
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    mp_pose = mp.solutions.pose
    mp_face_mesh = mp.solutions.face_mesh

    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
    pose = mp_pose.Pose( min_detection_confidence=0.5,min_tracking_confidence=0.5)

    # 모델 로드
    model = load_model('mp_hand_gesture')

    # 제스처 클래스 명 로드
    f = open('gesture.names', 'r')
    classNames = f.read().split('\n')
    f.close()


    audio = 'help.mp3'
    tts_kr = gTTS(text='안녕하세요', lang='ko')
    tts_kr.save(audio)
    #playsound(audio)

    prior_point = None
    movement_vector = np.zeros(2)
    cnt = 0
    cnt_stop = 7
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, image = cap.read()
        img_h, img_w, img_c = image.shape

        # Flip the image horizontally for a later selfie-view display
        # Also convert the color space from BGR to RGB
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

        # To improve performance
        image.flags.writeable = False

        # Get the result
        face_result = face_mesh.process(image)
        hand_result = hands.process(image)
        pose_result = pose.process(image)



        # To improve performance
        image.flags.writeable = True


        face_3d = []
        face_2d = []

        className = ''


        if face_result.multi_face_landmarks:
            for face_landmarks in face_result.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                        if idx == 1:
                            nose_2d = (lm.x * img_w, lm.y * img_h)
                            nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 8000)
                            #print(lm)

                        x, y = int(lm.x * img_w), int(lm.y * img_h)

                        # Get the 2D Coordinates
                        face_2d.append([x, y])

                        # Get the 3D Coordinates
                        face_3d.append([x, y, lm.z])

                        # Convert it to the NumPy array
                face_2d = np.array(face_2d, dtype=np.float64)

                # Convert it to the NumPy array
                face_3d = np.array(face_3d, dtype=np.float64)

                # The camera matrix
                focal_length = 1 * img_w

                cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                       [0, focal_length, img_w / 2],
                                       [0, 0, 1]])

                # The Distance Matrix
                dist_matrix = np.zeros((4, 1), dtype=np.float64)

                # Solve PnP
                success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

                # Get rotational matrix
                rmat, jac = cv2.Rodrigues(rot_vec)

                # Get angles
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                # Get the y rotation degree
                x = angles[0] * 360
                y = angles[1] * 360

                # print(y)

                # See where the user's head tilting
                if y < -10:
                    text = "Looking Left"
                elif y > 10:
                    text = "Looking Right"
                elif x < -10:
                    text = "Looking Down"
                else:
                    text = "Forward"

                # Display the nose direction
                nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

                p1 = (int(nose_2d[0]), int(nose_2d[1]))
                p2 = (int(nose_3d_projection[0][0][0]), int(nose_3d_projection[0][0][1]))

                cv2.line(image, p1, p2, (255, 0, 0), 2)

                # Add the text on the image
               # cv2.putText(image, text, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


        present_point = None
        if hand_result.multi_hand_landmarks:
            landmarks = []
            #if prior_hand_lms is not None:
                #print(hand_result.multi_hand_landmarks.landmark[8])
            for hand_landmarks in hand_result.multi_hand_landmarks:
                for idx, lm in enumerate(hand_landmarks.landmark):
                    # print(id, lm)
                    lmx = int(lm.x * img_h)
                    lmy = int(lm.y * img_w)

                    landmarks.append([lmx, lmy])
                    if idx  == 8:
                        present_point = np.array([lmx , lmy])
                        #print(present_point)

                if prior_point is not None:
                    v = present_point - prior_point
                    n = np.linalg.norm(v)
                    #print(v)
                    #print(n)
                    if n < 110:
                        movement_vector += v
                    '''
                    if v[0] > 10:
                        text = " right "
                    if v[0] < 10:
                        text = " left "
                    if v[1] > 10:
                        text = " up "
                    if v[0] > 10:
                        text = " right "
                    '''
                prior_point = present_point
                # Predict gesture
                prediction = model.predict([landmarks])
                # print(prediction)
                classID = np.argmax(prediction)
                className = classNames[classID]
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            prior_hand_lms = hand_result.multi_hand_landmarks

            # show the prediction on the frame

        # 포즈 주석을 이미지 위에 그립니다.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(image, pose_result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.putText(image, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow('output', image)
        cnt += 1
        if cnt == cnt_stop:
            print(movement_vector)
            mx , my = movement_vector
            if abs(mx) >= abs(my):
                if mx > 30:
                    print('move right')
                elif mx < -30:
                    print('move left')
            else:
                if my > 30:
                    print('move down')
                elif my < -30:
                    print('move up')

            cnt = 0
            movement_vector = np.zeros(2)
            prior_point = None
            cv2.waitKey()

        if cv2.waitKey(5) & 0xFF == 27:
            break

        '''
        with mp.solutions.hands.Hands(
                model_complexity=0,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as hands:
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            hand_result = hands.process(image)

            # 이미지에 손 주석을 그립니다.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if hand_result.multi_hand_landmarks:
                for hand_landmarks in face_result.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        with mp_pose.Pose(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as pose:
            # 필요에 따라 성능 향상을 위해 이미지 작성을 불가능함으로 기본 설정합니다.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            face_result = pose.process(image)

            # 포즈 주석을 이미지 위에 그립니다.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(image, face_result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow('Head Pose Estimation', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break
        '''
    cap.release()
if __name__ == "__main__":
    main()