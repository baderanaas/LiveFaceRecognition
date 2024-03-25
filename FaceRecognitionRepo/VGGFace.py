import numpy as np
import cv2
from joblib import load
from numpy import expand_dims
from cv2 import resize, INTER_CUBIC
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import ZeroPadding2D, Convolution2D, MaxPooling2D, Flatten, Activation, Dropout
from tensorflow.keras.preprocessing.image import img_to_array

class VGGFace:
    def __init__(self, weights_path="vgg_face_weights.h5", scaler_path="scaler.joblib", pca_path="pca_model.joblib", clf_path="SVC.joblib"):
        self.model = self._create_model()
        self.model.load_weights(weights_path)
        self.model = Model(inputs=self.model.layers[0].input, outputs=self.model.layers[-2].output)
        self.scaler = load(scaler_path)
        self.pca = load(pca_path)
        self.clf = load(clf_path)

    def _create_model(self):
        model = Sequential()
        model.add(ZeroPadding2D((1, 1), input_shape=(224, 224, 3)))
        model.add(Convolution2D(64, (3, 3), activation="relu"))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(64, (3, 3), activation="relu"))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(128, (3, 3), activation="relu"))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(128, (3, 3), activation="relu"))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, (3, 3), activation="relu"))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, (3, 3), activation="relu"))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, (3, 3), activation="relu"))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, (3, 3), activation="relu"))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, (3, 3), activation="relu"))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, (3, 3), activation="relu"))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, (3, 3), activation="relu"))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, (3, 3), activation="relu"))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, (3, 3), activation="relu"))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(Convolution2D(4096, (7, 7), activation="relu"))
        model.add(Dropout(0.5))
        model.add(Convolution2D(4096, (1, 1), activation="relu"))
        model.add(Dropout(0.5))
        model.add(Convolution2D(2622, (1, 1)))
        model.add(Flatten())
        model.add(Activation("softmax"))
        return model

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

# Example usage:
vgg_face = VGGFace()
# Assuming you have an image stored in 'image_path'
image = cv2.imread('image_path')
results, predictions = vgg_face.face_recognition(image)
print(results, predictions)
