import cv2
from FaceRecognitionRepo.VGGFace import VGGFace
import numpy as np
from facenet_pytorch import MTCNN
import os
import pytz
from datetime import datetime
from dotenv import load_dotenv
from pymongo import MongoClient, errors


def updating_lists(room):
    month = int(datetime.now().strftime("%m"))

    if month <= 12 and month >= 9:
        semestre = 1
    elif month >= 1 and month <= 6:
        semestre = 2
    else:
        semestre = 0

    today = datetime.now().strftime("%A")
    lectures = []
    lt = []
    teacher_id = None

    try:
        query = {"Room": room, "Day": today, "Semester": semestre}
        sort_order = [("StartTime", 1)]

        result = timetable.find(query).sort(sort_order)
        for doc in result:
            lectures.append(doc)

        if len(lectures) > 0:
            students_query = {"class_id": lectures[0]["class_id"]}
            students = student.find(students_query)
            for st in students:
                lt.append(str(st["Student_id"]))

            teacher_query = {"_id": lectures[0]["teacher_id"]}
            teachr = teacher.find_one(teacher_query)
            if teachr:
                teacher_id = teachr["Teacher_id"]

    except errors.PyMongoError as e:
        print(f"Database error: {e}")
        return [], [], today, semestre, teacher_id

    return lectures, lt, today, semestre, teacher_id


load_dotenv()

client_id = os.getenv("PPP_KEY")
client = MongoClient(client_id)
db = client["FaceDetection"]
attendance = db["Attendance"]
attendance_teacher = db["Attendance_Teacher"]
timetable = db["Time_table"]
teacher = db["Teacher"]
subject = db["Subject"]
classe = db["Class"]
student = db["Student"]

# room = input()
room = "212"

mtcnn = MTCNN(
    image_size=160, margin=14, min_face_size=20, device="cpu", post_process=False
)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1200)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1600)

data_folder_path = "data/ids"
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
teacher_records = {}
other = 0
font_scale = 1
font_thickness = 2

lectures, lt, day, semestre, teacher_id = updating_lists(room)

while True:
    now = datetime.now().time()
    today = datetime.now().strftime("%A")
    month = int(datetime.now().strftime("%m"))

    if day != today:
        records = {}
        lectures, lt, day, semestre, teacher_id = updating_lists(room)

    if len(lectures) == 0:
        continue

    if now >= datetime.strptime(lectures[0]["EndTime"], "%H:%M").time():
        records = {}
        teacher_records = {}
        lectures.pop(0)
        continue

    if now < datetime.strptime(lectures[0]["StartTime"], "%H:%M").time():
        continue

    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        continue

    try:
        frame = cv2.resize(frame, (1600, 1200), interpolation=cv2.INTER_CUBIC)
        frame_show = frame.copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.GaussianBlur(frame, ksize=(3, 3), sigmaX=0)
        frame_face = frame.copy()
        frame_face = cv2.resize(frame_face, (640, 640), interpolation=cv2.INTER_CUBIC)

        boxes, probs = mtcnn.detect(frame_face, landmarks=False)

        face_results = []

        if not probs is None and probs.all() > 0.6:
            for x1, y1, x2, y2 in boxes:
                x1, x2, y1, y2 = (
                    int(x1) * 1600 // 640,
                    int(x2) * 1600 // 640,
                    int(y1) * 1200 // 640,
                    int(y2) * 1200 // 640,
                )
                roi = frame[y1:y2, x1:x2]
                if roi.size == 0:
                    continue

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

                tunisia_tz = pytz.timezone("Africa/Tunis")
                curr_time = datetime.now(tunisia_tz).isoformat()
                face_result["date"] = datetime.fromisoformat(curr_time)
                face_results.append(face_result)
    except Exception as e:
        print(f"Error processing frame: {e}")
        continue

    if len(face_results) == 0:
        continue

    try:
        for face_result in face_results:
            find_query = {
                "id": face_result["id"],
                "timetable": lectures[0]["_id"],
                "date": datetime.now().strftime("%Y-%m-%d"),
            }
            found_st = attendance.find_one(find_query)

            if face_result["id"] in lt and found_st == None:

                if float(face_result["confidence"]) > 0.9:
                    records[face_result["id"]] = {
                        "day": face_result["date"],
                        "timetable": lectures[0]["_id"],
                        "date": datetime.now().strftime("%Y-%m-%d"),
                    }
                    lt.remove(face_result["id"])

            find_query = {
                "id": face_result["id"],
                "timetable": lectures[0]["_id"],
                "date": datetime.now().strftime("%Y-%m-%d"),
            }
            found_t = attendance_teacher.find_one(find_query)

            if face_result["id"] == teacher_id and found_t == None:
                if float(face_result["confidence"]) > 0.9:
                    teacher_records[face_result["id"]] = {
                        "day": face_result["date"],
                        "timetable": lectures[0]["_id"],
                        "date": datetime.now().strftime("%Y-%m-%d"),
                    }

        for record_id, record in records.items():
            record_data = {"id": record_id, **record}
            result = attendance.update_one(
                {
                    "$and": [
                        {"id": record_id},
                        {"day": record["day"]},
                        {"timetable": record["timetable"]},
                        {"date": record["date"]},
                    ]
                },
                {"$setOnInsert": record_data},
                upsert=True,
            )

        for record_id, record in teacher_records.items():
            record_data = {"id": record_id, **record}
            result = attendance_teacher.update_one(
                {
                    "$and": [
                        {"id": record_id},
                        {"day": record["day"]},
                        {"timetable": record["timetable"]},
                        {"date": record["date"]},
                    ]
                },
                {"$setOnInsert": record_data},
                upsert=True,
            )

    except errors.PyMongoError as e:
        print(f"Error updating attendance: {e}")

    cv2.imshow("Video", frame_show)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
