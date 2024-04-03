import cv2
from mtcnn.mtcnn import MTCNN
from concurrent.futures import ThreadPoolExecutor
import time
import os
import pygame

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

mtcnn_detector = MTCNN(scale_factor=0.7, min_face_size=20)

face_count = 0
studentID = input("Enter your student ID: ")

folderPath = f"data/{studentID}"
os.makedirs(folderPath, exist_ok=True)

# Initialize pygame mixer
pygame.mixer.init()

# Load the sound file
sound_file = "data/sound.wav"
pygame.mixer.music.load(sound_file)

# Play the sound
pygame.mixer.music.play()

# Allow some time for the sound to finish playing
while pygame.mixer.music.get_busy():
    time.sleep(2)

start_time = time.time()

while True:
    ret, frame = cap.read()

    cv2.imshow("Live Face Recognition", frame)
    if cv2.waitKey(1) == ord("q") or face_count >= 200:
        break

    # Resize the frame
    imgB = cv2.resize(frame, (0, 0), None, 1.5, 1.5)
    imgB = cv2.cvtColor(imgB, cv2.COLOR_BGR2RGB)

    def detect_face(imgB):
        return mtcnn_detector.detect_faces(imgB)

    with ThreadPoolExecutor() as executor:
        future = executor.submit(detect_face, imgB)
        data = future.result()

    # Find the faces in the current frame
    if data:

        bbox = [
            int(data[0]["box"][0] / 1.5),
            int(data[0]["box"][1] / 1.5),
            int(data[0]["box"][2] / 1.5),
            int(data[0]["box"][3] / 1.5),
        ]

        # Extract and save the face region
        x, y, w, h = bbox
        face_region = frame[y : y + h, x : x + w]
        print(face_count)
        cv2.imwrite(os.path.join(folderPath, f"face_{face_count}.jpg"), face_region)
        face_count += 1
        start_time = time.time()
    else:
        continue

    data.clear()

cap.release()
cv2.destroyAllWindows()
