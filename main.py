import cv2
from FaceRecognitionRepo.VGGFace import VGGFace
import numpy as np
from facenet_pytorch import MTCNN
import os
import time
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

client_id = os.getenv("ANAS_KEY")

client = MongoClient(client_id)
db = client["PPP"]
collection = db["Attendance"]

mtcnn = MTCNN(
    image_size=160, margin=14, min_face_size=20, device="cpu", post_process=False
)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1200)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1600)

classes = {"2100270": 0, "2100432": 1}


def ImageClass(n):
    for x, y in classes.items():
        if n == y:
            return x


vgg = VGGFace()
records = {}
other = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (1600, 1200), interpolation=cv2.INTER_CUBIC)
    frame = cv2.GaussianBlur(frame, ksize=(3, 3), sigmaX=0)
    frame_face = frame.copy()
    frame_face = cv2.resize(frame_face, (640, 640), interpolation=cv2.INTER_CUBIC)
    boxes, probs = mtcnn.detect(frame_face, landmarks=False)

    face_results = []

    if not probs.all() == None and probs.all() > 0.6:
        for x1, y1, x2, y2 in boxes:
            x1, x2, y1, y2 = (
                int(x1) * 1600 // 640,
                int(x2) * 1600 // 640,
                int(y1) * 1200 // 640,
                int(y2) * 1200 // 640,
            )
            roi = frame[y1:y2, x1:x2]
            result, y_predict = vgg.face_recognition(roi)
            face_result = dict()
            if len(result) > 1:
                face_result["id"] = ImageClass(result[0])
                face_result["confidence"] = str(np.round(y_predict[result[0]], 2))
            elif len(result) == 1:
                face_result["id"] = ImageClass(result[0])
                face_result["confidence"] = str(np.round(y_predict[result[0]], 2))
            secs = time.time()
            current_time_struct = time.localtime()
            formatted_time = time.strftime("%H:%M:%S", current_time_struct)
            face_result["time"] = formatted_time
            face_result["day"] = current_time_struct.tm_mday
            face_result["t"] = secs
            face_results.append(face_result)

    for face_result in face_results:
        if float(face_result["confidence"]) > 0.9:
            if face_result["id"] not in records.keys():
                records[face_result["id"]] = {
                    "time": face_result["time"],
                    "day": face_result["day"],
                    "t": face_result["t"],
                }
            else:
                if time.time() - records[face_result["id"]]["t"] > 1800:
                    records[face_result["id"]] = {
                        "time": face_result["time"],
                        "day": face_result["day"],
                        "t": face_result["t"],
                    }
    print(records)

    for record_id, record in records.items():
        record_data = {"id": record_id, **record}
        result = collection.update_one(
            {
                "$and": [
                    {"id": record_id},
                    {"time": record["time"]},
                    {"day": record["day"]},
                ]
            },
            {"$setOnInsert": record_data},
            upsert=True,
        )

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
