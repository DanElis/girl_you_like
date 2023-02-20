import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from keras import backend as K

from face_detector import MPFaceDetection
from facial_atribute_predict import FacialAttributeClassifier

sess = tf.Session()
with tf.device('/cpu:0'):
    FBP_model = tf.saved_model.load(sess=sess,
                                    tags=[tf.saved_model.tag_constants.SERVING],
                                    export_dir='model2-tf')

CASCADE = "Face_cascade.xml"
# FACE_CASCADE = cv2.CascadeClassifier(CASCADE)
face_detector = MPFaceDetection()
facial_attribute_classifier = FacialAttributeClassifier()


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


def face_rating(image, type_predict):
    ratings = [-1]
    processed_faces = extract_faces(image)
    if (type_predict == "clear" and len(processed_faces) > 1) or len(processed_faces) == 0:
        return -1

    for face in processed_faces:
        # Apply the neural network to predict face beauty.
        gender, age = facial_attribute_classifier.gender_age(face)
        print(gender, age)
        if age not in ['(15-20)', '(25-32)', '(38-43)']:
            continue
        if gender != 'Female':
            continue
        x = tf.get_default_graph().get_tensor_by_name('resnet50_input:0')
        y = tf.get_default_graph().get_tensor_by_name('dense_1/BiasAdd:0')
        pred = sess.run(y, feed_dict={x: np.expand_dims(face, 0)})
        ratings.append(pred[0][0])
    return max(ratings)


def is_it_like(max_rating):
    liked = False
    if max_rating > 2.65:
        liked = True
    return liked


def predict_like(image, type_predict):
    max_rating = face_rating(image, type_predict)
    return is_it_like(max_rating)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-path", dest="img_path", type=str, default='/mnt/c/inst_proj/main/#singapore')
    args = parser.parse_args()
    dir_path = args.img_path
    results = []
    dir_path = Path(dir_path)
    for img_name in os.listdir(dir_path):
        print(img_name)
        if img_name.split('.')[-1].lower() not in ['.jpeg', '.png', 'jpg']:
            continue

        img = Image.open(dir_path / img_name)
        img = np.array(img)
        results.append((img_name, predict_like(img, 'all')))

    # json_object = json.dumps({'results': results}, indent=4)
    # with open('/mnt/c/inst_proj/main/singapore_results.json', 'w') as f:
    #     f.write(json_object)
