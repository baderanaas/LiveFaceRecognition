import numpy as np
import cv2
from joblib import load
from numpy import expand_dims
from cv2 import resize, INTER_CUBIC
import tensorflow
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model


class VGGFace:
    def __init__(
        self,
        model_path="D:\INSAT\RT3\PPP\LiveFaceRecognition\FaceRecognitionRepo\models\model.h5",
        scaler_path="D:\INSAT\RT3\PPP\LiveFaceRecognition\FaceRecognitionRepo\models\scaler.joblib",
        pca_path="D:\INSAT\RT3\PPP\LiveFaceRecognition\FaceRecognitionRepo\models\pca_model.joblib",
        clf_path="D:\INSAT\RT3\PPP\LiveFaceRecognition\FaceRecognitionRepo\models\SVC.joblib",
    ):
        self.model = load_model(model_path)
        self.model = Model(
            inputs=self.model.layers[0].input, outputs=self.model.layers[-2].output
        )
        self.scaler = load(scaler_path)
        self.pca = load(pca_path)
        self.clf = load(clf_path)

    def preprocess_image(self, img):
        img = img_to_array(img)
        img = img / 255.0
        img = expand_dims(img, axis=0)
        return img

    def face_recognition(self, roi):
        roi = resize(roi, dsize=(224, 224), interpolation=INTER_CUBIC)
        roi = self.preprocess_image(roi)
        embedding_vector = self.model.predict(roi)[0]

        embedding_vector = self.scaler.transform(embedding_vector.reshape(1, -1))
        embedding_vector_pca = self.pca.transform(embedding_vector)
        result1 = self.clf.predict(embedding_vector_pca)[0]

        y_predict = self.clf.predict_proba(embedding_vector_pca)[0]

        result = np.where(y_predict > 0.3)[0]

        return result, y_predict
