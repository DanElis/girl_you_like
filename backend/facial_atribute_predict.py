# import os
# from pathlib import Path

import cv2
# from dotenv import load_dotenv

# env_path = Path('./') / '.env'
# load_dotenv(dotenv_path=env_path)

# logger = logging.getLogger()
# logger.setLevel(logging.INFO)
# import pdb

# Load path from .env
# faceProto = os.getenv("FACEDETECTOR")
# faceModel = os.getenv("FACEMODEL")
ageProto = '../age/age_deploy.prototxt'
ageModel = '../age/age_net.caffemodel'
genderProto = '../gender/gender_deploy.prototxt'
genderModel = '../gender/gender_net.caffemodel'


class FacialAttributeClassifier:
    def __init__(self):
        # Load face detection model
        # self.face_net = cv2.dnn.readNet(faceModel, faceProto)
        # Load age detection model
        self.age_net = cv2.dnn.readNet(ageModel, ageProto)
        # Load gender detection model
        self.gender_net = cv2.dnn.readNet(genderModel, genderProto)

#     def get_face_box(self, image, conf_threshold=0.7):
#         image = image.copy()
#         image_height = image.shape[0]
#         image_width = image.shape[1]
#         # print(image.shape)
#         blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123], True, False)
# #        print(blob)
# #        print(blob.shape)
# #        blob = np.squeeze(blob)
#         self.face_net.setInput(blob)
#         detections = self.face_net.forward()
#         face_boxes = []
#         for i in range(detections.shape[2]):
#             confidence = detections[0, 0, i, 2]
#             if confidence > conf_threshold:
#                 x1 = int(detections[0, 0, i, 3] * image_width)
#                 y1 = int(detections[0, 0, i, 4] * image_height)
#                 x2 = int(detections[0, 0, i, 5] * image_width)
#                 y2 = int(detections[0, 0, i, 6] * image_height)
#                 face_boxes.append([x1, y1, x2, y2])
#                 cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), int(round(image_height / 150)), 8)
#         return image, face_boxes

    def gender_age(self, face):
        model_mean_values = (78.4263377603, 87.7689143744, 114.895847746)
        age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
        gender_list = ['Male', 'Female']

        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), model_mean_values, swapRB=False)

        # Predict the gender
        self.gender_net.setInput(blob)
        gender_preds = self.gender_net.forward()
        gender = gender_list[gender_preds[0].argmax()]
        # Predict the age
        self.age_net.setInput(blob)
        age_preds = self.age_net.forward()
        age = age_list[age_preds[0].argmax()]
        # Return
        return gender, age

    # def predict(self, image):
    #     symbol = lightened_moon_feature(num_classes=40, use_fuse=True)
    #     devs = mx.cpu()
    #     _, arg_params, aux_params = mx.model.load_checkpoint('', 82)
    #
    #     ''' Loading Image from directory and writing attributes into .txt file'''
    #     # img_dir = os.path.join(pathImg)
    #     # if os.path.exists(img_dir):
    #     #     names = os.listdir(pathImg)
    #     #     img_paths = [name for name in names]
    #     #     for imge in range(4005, 4005 + len(names)):
    #     #
    #     #         imge = "{:06d}.jpg".format(imge)
    #     #         path = pathImg + str(imge)
    #     #         print("Image Path", path)
    #             # read img and drat face rect
    #             # image = cv2.imread(path)
    #             # img = cv2.imread(path, -1)
    #
    #     result_img, face_boxes = self.get_face_box(image)
    #     if not face_boxes:
    #         print("No face detected")
    #
    #     # Loop throuth the coordinates
    #     for face_box in face_boxes:
    #
    #         # print("#====Detected Age and Gender====#")
    #
    #         gender, age = self.gender_age(image, face_box)
    #
    #         left = face_box[0]
    #         width = face_box[2] - face_box[0]
    #         top = face_box[1]
    #         height = face_box[3] - face_box[1]
    #         right = face_box[2]
    #         bottom = face_box[3]
    #         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #         pad = [0.25, 0.25, 0.25, 0.25]
    #         left = int(max(0, left - width * float(pad[0])))
    #         top = int(max(0, top - height * float(pad[1])))
    #         right = int(min(gray.shape[1], right + width * float(pad[2])))
    #         bottom = int(min(gray.shape[0], bottom + height * float(pad[3])))
    #         gray = gray[left:right, top:bottom]
    #         # resizing image and increasing the image size
    #         gray = cv2.resize(gray, (128, 128)) / 255.0
    #         img = np.expand_dims(np.expand_dims(gray, axis=0), axis=0)
    #         # get image parameter from mxnet
    #         arg_params['data'] = mx.nd.array(img, devs)
    #         exector = symbol.bind(devs, arg_params, args_grad=None, grad_req="null", aux_states=aux_params)
    #         exector.forward(is_train=False)
    #         exector.outputs[0].wait_to_read()
    #         output = exector.outputs[0].asnumpy()
    #         # 40 facial attributes
    #         text = ["5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes", "Bald", "Bangs",
    #                 "Big_Lips", "Big_Nose",
    #                 "Black_Hair", "Blond_Hair", "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin",
    #                 "Eyeglasses", "Goatee",
    #                 "Gray_Hair", "Heavy_Makeup", "High_Cheekbones", "Male", "Mouth_Slightly_Open", "Mustache",
    #                 "Narrow_Eyes", "No_Beard",
    #                 "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline", "Rosy_Cheeks", "Sideburns",
    #                 "Smiling", "Straight_Hair",
    #                 "Wavy_Hair", "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace",
    #                 "Wearing_Necktie", "Young"]
    #
    #         # Predict the results
    #         # create a list based on the attributes generated.
    #         attr_dict = {}
    #         detected_attribute_list = []
    #         for i in range(40):
    #             attr = text[i].rjust(20)
    #             if output[0][i] < 0:
    #                 attr_dict[attr] = 'No'
    #             else:
    #                 attr_dict[attr] = 'Yes'
    #                 detected_attribute_list.append(text[i])
    #     return detected_attribute_list + [gender]
