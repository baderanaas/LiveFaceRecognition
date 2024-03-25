import cv2
from FaceRecognitionRepo import VGGFace    
import numpy as np
import matplotlib.pyplot as plt
from facenet_pytorch import MTCNN
import os

mtcnn = MTCNN(image_size=160, margin=14, min_face_size=20,device='cpu', post_process=False)

cap=cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1200)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1600)

size = (1600, 1200)

result_video = cv2.VideoWriter('Face.avi', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, size)

vgg = VGGFace()

other = 0
while True :
    ret, frame = cap.read()
    if not ret:
        break  
    frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    frame=cv2.resize(frame,(1600,1200),interpolation=cv2.INTER_CUBIC)
    frame=cv2.GaussianBlur(frame, ksize=(3,3), sigmaX=0)
    frame_face = frame.copy()
    frame_face=cv2.resize(frame_face,(640,640),interpolation=cv2.INTER_CUBIC)
    boxes, probs = mtcnn.detect(frame_face, landmarks=False)

    face_results = []

    if not probs.all() == None and probs.all() > 0.6:
        for x1, y1, x2, y2 in boxes:
            x1, x2, y1, y2 = int(x1) * 1600 // 640, int(x2) * 1600 // 640, int(y1) * 1200 // 640, int(y2) * 1200 // 640
            roi = frame[y1:y2, x1:x2]
            result, y_predict = vgg.Face_Recognition(roi)
            face_result = {}
            if len(result) > 1:
                face_result['label'] = result[0]
                face_result['confidence'] = str(np.round(y_predict[result[0]], 2))
            elif len(result) == 0:
                roi = cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)
                folderPath = f"data/other"
                os.makedirs(folderPath, exist_ok=True)
                cv2.imwrite(os.path.join(folderPath, f"face_{other}.jpg"), roi)
                face_result['label'] = 'Other'
                other += 1
            else:
                face_result['label'] = result
                face_result['confidence'] = str(np.round(y_predict[result[0]], 2))
            face_result['box'] = (x1, y1, x2, y2)
            face_results.append(face_result)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    result_video.write(frame)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

    print(face_results)

cap.release()
result_video.release()
cv2.destroyAllWindows() 