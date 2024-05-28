import cv2
from FaceRecognitionRepo.VGGFace import VGGFace
import numpy as np
from facenet_pytorch import MTCNN
import os
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
from pymongo import MongoClient


def updating_lists(room):
    today = datetime.now().strftime("%A")

    lectures = []

    query = {"Room": room, "Day": today}
    sort_order = [("StartTime", 1)]

    result = timetable.find(query).sort(sort_order)
    for doc in result:
        lectures.append(doc)

    students_query = {"class_id": lectures[0]["class_id"]}

    lt = []
    students = student.find(students_query)
    for st in students:
        lt.append(st["Student_id"])

    teacher_query = {"_id": lectures[0]["teacher_id"]}
    teachr = teacher.find_one(teacher_query)
    lt.append(teachr["CIN"])

    return lectures, lt, today


load_dotenv()

client_id = os.getenv("ANAS_KEY")

client = MongoClient(client_id)
db = client["FaceDetection"]
attendance = db["Attendance"]
timetable = db["Time_table"]
teacher = db["Teacher"]
subject = db["Subject"]
classe = db["Class"]
student = db["Student"]

# room = input("Enter the room number: ")
room = "111"

mtcnn = MTCNN(
    image_size=160, margin=14, min_face_size=20, device="cpu", post_process=False
)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1200)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1600)

data_folder_path = "data"
folder_names = [
    name
    for name in os.listdir(data_folder_path)
    if os.path.isdir(os.path.join(data_folder_path, name))
]

classes = {}

for index, folder_name in enumerate(folder_names):
    classes[folder_name] = index


def ImageClass(n):
    for x, y in classes.items():
        if n == y:
            return x


vgg = VGGFace()
records = {}
other = 0
font_scale = 1
font_thickness = 2

lectures, lt, day = updating_lists(room)


while True:
    now = datetime.now().time()
    today = datetime.now().strftime("%A")

    if day != today:
        records = {}
        lectures, lt, day = updating_lists(room)

    if len(lectures) == 0:
        continue

    while now >= datetime.strptime(lectures[0]["EndTime"], "%H:%M").time():
        records = {}
        lectures.pop(0)
        if not lectures:
            break

    if len(lectures) == 0:
        continue

    if now < datetime.strptime(lectures[0]["StartTime"], "%H:%M").time():
        continue

    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (1600, 1200), interpolation=cv2.INTER_CUBIC)
    frame_show = frame.copy()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
            if len(result) >= 1:
                face_result["id"] = ImageClass(result[0])
                face_result["confidence"] = str(np.round(y_predict[result[0]], 2))

                cv2.rectangle(frame_show, (x1, y1), (x2, y2), (0, 255, 0), 2)
                text_size = cv2.getTextSize(
                    face_result["id"],
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    font_thickness,
                )[0]
                text_x = x1 + 10
                text_y = y1 + text_size[1] + 10
                cv2.putText(
                    frame_show,
                    face_result["id"],
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (0, 255, 0),
                    font_thickness,
                )

            current_time_struct = time.localtime()
            formatted_time = time.strftime("%Y-%m-%dT%H:%M:%S", current_time_struct)
            face_result["date"] = formatted_time
            face_results.append(face_result)

    if len(face_results) == 0:
        continue

    for face_result in face_results:
        if face_result["id"] in lt:
            if float(face_result["confidence"]) > 0.9:
                if face_result["id"] not in records.keys():
                    records[face_result["id"]] = {
                        "day": face_result["date"],
                    }
                    lt.remove(face_result["id"])

    # print(records)

    for record_id, record in records.items():
        record_data = {"id": record_id, **record}
        result = attendance.update_one(
            {
                "$and": [
                    {"id": record_id},
                    {"day": record["day"]},
                ]
            },
            {"$setOnInsert": record_data},
            upsert=True,
        )

    cv2.imshow("Video", frame_show)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
