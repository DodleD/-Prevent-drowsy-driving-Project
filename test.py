import cv2, dlib
import numpy as np
import datetime
from imutils import face_utils
from keras.models import load_model
import pygame
voice_file = "warning.mp3" #경고음 파일
pygame.mixer.init()
IMG_SIZE = (34, 26)

start = datetime.datetime.now()
end = datetime.datetime.now()
close_eye = False

detector = dlib.get_frontal_face_detector() #dlib 얼굴 인식 라이브러리
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat') 

model = load_model('models/model.h5')
model.summary()

def check_drowsiness(left,right):
    left = left.split()[0]
    right = right.split()[0]    
    if left == '-' and right == '-':                
        return True
    else:
        return False

def crop_eye(img, eye_points): #얼굴에서 눈 부분만 cropping 하는 함수
  x1, y1 = np.amin(eye_points, axis=0)
  x2, y2 = np.amax(eye_points, axis=0)
  cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

  w = (x2 - x1) * 1.2
  h = w * IMG_SIZE[1] / IMG_SIZE[0]

  margin_x, margin_y = w / 2, h / 2

  min_x, min_y = int(cx - margin_x), int(cy - margin_y)
  max_x, max_y = int(cx + margin_x), int(cy + margin_y)

  eye_rect = np.rint([min_x, min_y, max_x, max_y]).astype(np.int)

  eye_img = gray[eye_rect[1]:eye_rect[3], eye_rect[0]:eye_rect[2]]

  return eye_img, eye_rect

# main
# cap = cv2.VideoCapture('videos/2.mp4')
cap = cv2.VideoCapture(0)

while cap.isOpened():  
  ret, img_ori = cap.read()  
  if not ret:
    break

  if close_eye:
    end = datetime.datetime.now()
    # 눈감은 뒤, 1초지났는지 체크하기 위함.
    check_time = 1 #1초
    time_diff = end - start 
    if time_diff.seconds >= check_time:
        if pygame.mixer.music.get_busy() == False:            
            pygame.mixer.music.load(voice_file)
            pygame.mixer.music.set_volume(1/2)
            pygame.mixer.music.play(-1)    
        print("warning")     
  else:
    if pygame.mixer.music.get_busy():
        pygame.mixer.music.stop()
  img_ori = cv2.resize(img_ori, dsize=(0, 0), fx=0.5, fy=0.5)

  img = img_ori.copy()
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  #gray 이미지에서 얼굴 인식 
  faces = detector(gray)

  for face in faces:
    shapes = predictor(gray, face)
    shapes = face_utils.shape_to_np(shapes)

    #왼쪽, 오른쪽 눈 crop
    eye_img_l, eye_rect_l = crop_eye(gray, eye_points=shapes[36:42])
    eye_img_r, eye_rect_r = crop_eye(gray, eye_points=shapes[42:48])

    eye_img_l = cv2.resize(eye_img_l, dsize=IMG_SIZE)
    eye_img_r = cv2.resize(eye_img_r, dsize=IMG_SIZE)
    eye_img_r = cv2.flip(eye_img_r, flipCode=1)

    eye_input_l = eye_img_l.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32) / 255.
    eye_input_r = eye_img_r.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32) / 255.

    #사전에 학습된 모델을 활용하여 눈의 졸음 여부 파악(Classification -> label : 졸음 o/ 졸음 x) 
    pred_l = model.predict(eye_input_l) 
    pred_r = model.predict(eye_input_r)

    # visualize (졸음 여부 상태 문구 : - 가 앞에 붙어있으면 졸음 o)
    state_l = 'O %.1f' if pred_l > 0.1 else '- %.1f'
    state_r = 'O %.1f' if pred_r > 0.1 else '- %.1f'

    state_l = state_l % pred_l
    state_r = state_r % pred_r    

    #문구에 -가 들어감 -> 졸음 o / 안들어감 -> 졸음 x
    result = check_drowsiness(state_l, state_r)

    if result and close_eye == False:
        start = datetime.datetime.now()
        close_eye = True
    elif result == False:
        close_eye = False
    cv2.rectangle(img, pt1=tuple(eye_rect_l[0:2]), pt2=tuple(eye_rect_l[2:4]), color=(255,255,255), thickness=2)
    cv2.rectangle(img, pt1=tuple(eye_rect_r[0:2]), pt2=tuple(eye_rect_r[2:4]), color=(255,255,255), thickness=2)

    cv2.putText(img, state_l, tuple(eye_rect_l[0:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(img, state_r, tuple(eye_rect_r[0:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

  cv2.imshow('result', img)
  if cv2.waitKey(1) == ord('q'):
    break
