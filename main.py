import cv2
import mediapipe as mp
import numpy as np
import time

from tensorflow.keras.models import load_model
from collections import deque
from statistics import mean
from gtts import gTTS
from playsound import playsound

def unit_vector(vector):
    """ 벡터의 단위 벡터 리턴 """
    return vector / np.linalg.norm(vector)
def angle_between(v1, v2):
    '''두 벡터의 각도를 구함 '''
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
def split_vector(v1):
    ''' 3차원 벡터를 하나의 성분을 0으로 만들어 2차원 벡터로 만든 뒤 xz 성분 벡터와 yz 성분 벡터로 리턴 '''
    v1_xz = np.array([v1[0],0,v1[2]])
    v1_yz = np.array([0,v1[1],v1[2]])
    return [v1_xz, v1_yz]
def get_xz_yz_angle(v1, v2):
    '''두 벡터의 각도를 구함 '''
    sv1 = split_vector(v1)
    sv2 = split_vector(v2)
    xz_angle= angle_between(sv1[0], sv2[0])
    yz_angle = angle_between(sv1[1], sv2[1])
    return np.array([ xz_angle, yz_angle ])
def main():
    #몇 프레임을 움직임 계산을 할 때 사용할 지 결정하는 변수
    hand_brkt = 7
    nose_brkt = 4

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

    #안내문 생성
    audio = 'help.mp3'
    tts_kr = gTTS(text='손을 완전히 편채 손바닥을 화면에 보이고 천천히 상하좌우로 움직여 주세요', lang='ko')
    tts_kr.save(audio)
    #playsound(audio)

    pri_hand_pt = None
    pri_nose_tip_pt = None
    pri_nose_udr_pt = None

    #nose_ang_vec = np.zeros(7)
    #계산을 위한 벡터와 큐
    hand_mvt_vec = np.zeros(2)
    hand_mvt_queue = deque([])
    nose_ang_queue = deque([])

    #설정된 프레임 단위 만큼 프레임을 포착해 움직임을 계산한다.
    timer_begin_pt = 0
    timer_end_pt = 0

    cap = cv2.VideoCapture(0)

    while cap.isOpened():

        flag = False

        success, image = cap.read()
        img_h, img_w, img_c = image.shape


        # 이미지를 flip하여 화면상에서 보기 편하게 만든다.
        # BGR 이미지를 RGB로 변환한다.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

        # 각각 결과를 저장하는 부분
        face_result = face_mesh.process(image)
        hand_result = hands.process(image)
        pose_result = pose.process(image)

        pre_nose_tip_pt = None
        pre_nose_udr_pt = None
        pre_hand_pt = None


        # To improve performance
        image.flags.writeable = True


        face_3d = []
        face_2d = []

        className = ''


        if face_result.multi_face_landmarks:
            for face_landmarks in face_result.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                        if idx == 1:
                            pre_nose_tip_pt = np.array([lm.x * img_w, lm.y * img_h, lm.z * 100])
                        if idx == 2:
                            pre_nose_udr_pt = np.array([lm.x * img_w, lm.y * img_h, lm.z * 100])


                        # Convert it to the NumPy array

                # 이전 포인트가 None이 아닌 경우
                if pri_nose_tip_pt is not None and pre_nose_udr_pt is not None:
                    # 움직임 벡터를 구함
                    pre_vec = pre_nose_tip_pt - pre_nose_udr_pt
                    pri_vec = pri_nose_tip_pt - pri_nose_udr_pt
                    xz_angle = get_xz_yz_angle(pre_vec, pri_vec)[0]
                    if xz_angle < 2:
                        nose_ang_queue.append(xz_angle)
                    #nose_ang_vec = np.append(get_xz_yz_angle(pre_vec, pri_vec)[0], nose_ang_vec)

                pri_nose_tip_pt = pre_nose_tip_pt
                pri_nose_udr_pt = pre_nose_udr_pt


        #손 움직임 인식 부분


        if hand_result.multi_hand_landmarks:
            landmarks = []
            #if prior_hand_lms is not None:
                #print(hand_result.multi_hand_landmarks.landmark[8])
            for hand_landmarks in hand_result.multi_hand_landmarks:
                for idx, lm in enumerate(hand_landmarks.landmark):
                    # print(id, lm)
                    lmx = int(lm.x * img_w)
                    lmy = int(lm.y * img_h)


                    landmarks.append([lmx, lmy])
                    if idx  == 0:
                        pre_hand_pt = np.array([lmx , lmy])
                        #print(present_point)
                        
                #이전 손 포인트가 None이 아닌 경우
                if pri_hand_pt is not None:
                    #움직임 벡터를 구함
                    v = pre_hand_pt - pri_hand_pt
                    #벡터의 크기를 구함
                    n = np.linalg.norm(v)
                    #print(v)
                    #print(n)
                    
                    #벡터의 크기가 너무 큰 경우 노이즈로 처리
                    if n < 100:
                        hand_mvt_queue.append(v)
                        hand_mvt_vec += v


                pri_hand_pt = pre_hand_pt
                # 제스처 예측
                prediction = model.predict([landmarks])
                # print(prediction)
                classID = np.argmax(prediction)
                className = classNames[classID]
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            prior_hand_lms = hand_result.multi_hand_landmarks

            # show the prediction on the frame
        cv2.putText(image, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2, cv2.LINE_AA)

        # 포즈 주석을 이미지 위에 표시.

        #mp_drawing.draw_landmarks(image, pose_result.pose_landmarks, mp_pose.POSE_CONNECTIONS)



        timer_end_pt =time.time()
        #얼굴 고개 회전 판정 계산 부분
        if len(nose_ang_queue) >= nose_brkt:
            m = mean(nose_ang_queue)
            text = ""
            # print(m)
            if m > 0.4:
                if timer_begin_pt == 0:
                    flag = True
                text = "move head"
                print(text)
            a = nose_ang_queue.popleft()
            cv2.putText(image, text, (10, 150), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2, cv2.LINE_AA)
        #손 움직임 계산 부분
        if len(hand_mvt_queue) >= hand_brkt:
            mx, my = hand_mvt_vec
            text =""
            if abs(mx) >= abs(my):
                if mx > 30:
                    text ='move right'
                    print(text)
                elif mx < -30:
                    text = 'move left'
                    print(text)
            else:
                if my > 30:
                    text = 'move down'
                    print(text)
                elif my < -30:
                    text = 'move up'
                    print(text)

            cv2.putText(image, text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2, cv2.LINE_AA)
            v = hand_mvt_queue.popleft()
            hand_mvt_vec -= v
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow('output', image)


        timer_result = timer_end_pt - timer_begin_pt
        if timer_result >= 7:
            timer_begin_pt = 0

        if cv2.waitKey(5) & 0xFF == ord('p'):
            print('waiting...')
            cv2.waitKey()
        if cv2.waitKey(5) & 0xFF == ord('q'):
            print('end program')
            break


        if flag == True:
            playsound(audio)
            timer_begin_pt = time.time()

    cap.release()
if __name__ == "__main__":
    main()