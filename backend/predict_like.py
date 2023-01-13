import argparse

import cv2
import keras.models
import numpy as np
from PIL import Image

FBP_model = keras.models.load_model('model2.h5')

CASCADE = "Face_cascade.xml"
FACE_CASCADE = cv2.CascadeClassifier(CASCADE)


def extract_faces(image):
    processed_images = []

    image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Minimum size of detected faces is set to 75x75 pixels.
    faces = FACE_CASCADE.detectMultiScale(image_grey, scaleFactor=1.16, minNeighbors=5, minSize=(75, 75), flags=0)

    for x, y, w, h in faces:
        sub_img = image[y - 15:y + h + 15, x - 15:x + w + 15]
        side = np.max(np.array([sub_img.shape[0], sub_img.shape[1]]))
        sub_image_padded = cv2.copyMakeBorder(sub_img, int(np.floor((side - sub_img.shape[1]) / 2)),
                                              int(np.ceil((side - sub_img.shape[1]) / 2)),
                                              int(np.floor((side - sub_img.shape[0]) / 2)),
                                              int(np.ceil((side - sub_img.shape[0]) / 2)), cv2.BORDER_CONSTANT)
        sub_image_resized = cv2.resize(src=sub_image_padded, dsize=(350, 350))
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
        pred = FBP_model.predict(face.reshape((1,)))
        ratings.append(pred[0][0])

    max_rating = max(ratings)
    # If the maximal rating received for a profile's photo is greater than 3, like the profile.
    if max_rating > 3:
        liked = True
    return liked


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-path", dest="img_path", type=str)
    args = parser.parse_args()
    img_path = args.img_path
    img = Image.open(img_path)
    liked = predict_like(img, "all")
    print(liked)
