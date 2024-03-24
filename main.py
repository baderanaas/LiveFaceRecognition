import cv2
import pickle
from mtcnn.mtcnn import MTCNN
from concurrent import futures
from concurrent.futures import ThreadPoolExecutor
import cvzone

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# load the encoding file
with open("encodings.p", "rb") as file:
    encodeListKnownWithIds = pickle.load(file)
encodeListKnown, studentIds = encodeListKnownWithIds

mtcnn_detector = MTCNN(scale_factor=0.7, min_face_size=20)


while True:
    ret, frame = cap.read()

    # Resize the frame
    imgB = cv2.resize(frame, (0, 0), None, 1.5, 1.5)
    imgB = cv2.cvtColor(imgB, cv2.COLOR_BGR2RGB)

    imgS = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    def detect_face(imgB):
        return mtcnn_detector.detect_faces(imgB)

    with ThreadPoolExecutor() as executor:
        future = executor.submit(detect_face, imgB)
        data = future.result()

    # Find the faces in the current frame
    faceCurrentFrame = []

    for item in data:
        bbox = [
            int(item["box"][0] / 1.5),
            int(item["box"][1] / 1.5),
            int(item["box"][2] / 1.5),
            int(item["box"][3] / 1.5)
        ]
        faceCurrentFrame.append(bbox)
        cvzone.cornerRect(frame, bbox, rt=0)
    print(faceCurrentFrame)

    cv2.imshow("Live Face Recognition", frame)
    cv2.waitKey(1)
