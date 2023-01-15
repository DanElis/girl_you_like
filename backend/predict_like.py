import argparse
import json
import os
from pathlib import Path

import cv2
import keras.models
import numpy as np
from PIL import Image
import tensorflow as tf

from face_detector import MPFaceDetection

with tf.device('/cpu:0'):
    FBP_model = keras.models.load_model('model2.h5')

CASCADE = "Face_cascade.xml"
# FACE_CASCADE = cv2.CascadeClassifier(CASCADE)
face_detector = MPFaceDetection()

def extract_faces(image):
    processed_images = []
    faces = face_detector(image)
    if faces is None:
        return []
    # image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Minimum size of detected faces is set to 75x75 pixels.
    # faces = FACE_CASCADE.detectMultiScale(image_grey, scaleFactor=0.95, minNeighbors=5, minSize=(75, 75), flags=0)

    for x, y, w, h in faces:
        sub_img = image[y - 15:y + h + 15, x - 15:x + w + 15]
        side = np.max(np.array([sub_img.shape[0], sub_img.shape[1]]))
        sub_image_padded = cv2.copyMakeBorder(sub_img, int(np.floor((side - sub_img.shape[1]) / 2)),
                                              int(np.ceil((side - sub_img.shape[1]) / 2)),
                                              int(np.floor((side - sub_img.shape[0]) / 2)),
                                              int(np.ceil((side - sub_img.shape[0]) / 2)), cv2.BORDER_CONSTANT)
        if sub_image_padded is None:
            sub_image_padded = sub_img
        try:
            sub_image_resized = cv2.resize(src=sub_image_padded, dsize=(350, 350))
        except:
            continue
        processed_images.append(sub_image_resized)
    return processed_images


def predict_like(image, type_predict):
    ratings = []
    processed_faces = extract_faces(image)
    liked = False
    if (type_predict == "clear" and len(processed_faces) > 1) or len(processed_faces) == 0:
        return liked

    for face in processed_faces:
        # Apply the neural network to predict face beauty.
        pred = FBP_model.predict(np.expand_dims(face, 0))
        ratings.append(pred[0][0])

    max_rating = max(ratings)

    if max_rating > 2.65:
        liked = True
    return liked


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-path", dest="img_path", type=str, default='/mnt/c/inst_proj/main/#singapore')
    args = parser.parse_args()
    dir_path = args.img_path
    results = []
    dir_path = Path(dir_path)
    for img_name in os.listdir(dir_path):
        if img_name.split('.')[-1].lower() not in ['.jpeg', '.png', 'jpg']:
            continue

        img = Image.open(dir_path / img_name)
        img = np.array(img)
        results.append((img_name, predict_like(img, 'all')))
    json_object = json.dumps({'results': results}, indent=4)

    with open('/mnt/c/inst_proj/main/singapore_results.json', 'w') as f:
        f.write(json_object)

