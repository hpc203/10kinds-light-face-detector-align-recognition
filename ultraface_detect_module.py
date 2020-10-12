import cv2
import numpy as np
from ultraface.utils import define_img_size, convert_locations_to_boxes, center_form_to_corner_form, predict

class ultraface():
    def __init__(self, framework='onnx', input_img_size = 320, threshold=0.7):
        if input_img_size == 320:
            if framework=='onnx':
                self.net = cv2.dnn.readNetFromONNX('ultraface/version-RFB-320_simplified.onnx')
            else:
                self.net = cv2.dnn.readNetFromCaffe('ultraface/RFB-320.prototxt', 'ultraface/RFB-320.caffemodel')
            self.input_size = (320, 240)
        else:
            self.net = cv2.dnn.readNetFromONNX('ultraface/version-slim-640.onnx')
            self.input_size = (640, 480)
        self.priors = define_img_size(self.input_size)
        self.threshold = threshold
        self.center_variance = 0.1
        self.size_variance = 0.2
        self.image_std = 128.0
    def detect(self, srcimg):
        # rect = cv2.resize(srcimg, (self.input_size[0], self.input_size[1]))
        # rect = cv2.cvtColor(rect, cv2.COLOR_BGR2RGB)
        # self.net.setInput(cv2.dnn.blobFromImage(rect, 1 / self.image_std, (self.input_size[0], self.input_size[1]), 127))

        # rect = cv2.cvtColor(srcimg, cv2.COLOR_BGR2RGB)
        # self.net.setInput(cv2.dnn.blobFromImage(rect, 1 / self.image_std, (self.input_size[0], self.input_size[1]), 127))

        rect = cv2.resize(srcimg, (self.input_size[0], self.input_size[1]))
        rect = cv2.cvtColor(rect, cv2.COLOR_BGR2RGB)
        self.net.setInput(cv2.dnn.blobFromImage(rect, scalefactor=1 / self.image_std, mean=127))

        boxes, scores = self.net.forward(["boxes", "scores"])
        boxes = np.expand_dims(np.reshape(boxes, (-1, 4)), axis=0)
        scores = np.expand_dims(np.reshape(scores, (-1, 2)), axis=0)
        boxes = convert_locations_to_boxes(boxes, self.priors, self.center_variance, self.size_variance)
        boxes = center_form_to_corner_form(boxes)
        boxes, labels, probs = predict(srcimg.shape[1], srcimg.shape[0], scores, boxes, self.threshold)
        drawimg, face_rois = srcimg.copy(), []
        for i in range(boxes.shape[0]):
            box = boxes[i, :]
            cv2.rectangle(drawimg, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), thickness=2)
            face_rois.append(srcimg[box[1]:box[3], box[0]:box[2]])
        return drawimg, face_rois
    def get_face(self, srcimg):
        # rect = cv2.resize(srcimg, (self.input_size[0], self.input_size[1]))
        # rect = cv2.cvtColor(rect, cv2.COLOR_BGR2RGB)
        # self.net.setInput(cv2.dnn.blobFromImage(rect, 1 / self.image_std, (self.input_size[0], self.input_size[1]), 127))

        # rect = cv2.cvtColor(srcimg, cv2.COLOR_BGR2RGB)
        # self.net.setInput(cv2.dnn.blobFromImage(rect, 1 / self.image_std, (self.input_size[0], self.input_size[1]), 127))

        rect = cv2.resize(srcimg, (self.input_size[0], self.input_size[1]))
        rect = cv2.cvtColor(rect, cv2.COLOR_BGR2RGB)
        self.net.setInput(cv2.dnn.blobFromImage(rect, scalefactor=1 / self.image_std, mean=127))

        boxes, scores = self.net.forward(["boxes", "scores"])
        boxes = np.expand_dims(np.reshape(boxes, (-1, 4)), axis=0)
        scores = np.expand_dims(np.reshape(scores, (-1, 2)), axis=0)
        boxes = convert_locations_to_boxes(boxes, self.priors, self.center_variance, self.size_variance)
        boxes = center_form_to_corner_form(boxes)
        boxes, labels, probs = predict(srcimg.shape[1], srcimg.shape[0], scores, boxes, self.threshold)
        face_rois = []
        for i in range(boxes.shape[0]):
            face_rois.append(srcimg[boxes[i, 1]:boxes[i, 3], boxes[i, 0]:boxes[i, 2]])
        return boxes.tolist(), face_rois

if __name__ == '__main__':
    ultraface_detect = ultraface(framework='onnx', input_img_size=640)
    imgpath = 'selfie.jpg'
    srcimg = cv2.imread(imgpath)
    drawimg, face_rois = ultraface_detect.detect(srcimg)

    # _, face_rois = ultraface_detect.get_face(srcimg)
    # for i, face in enumerate(face_rois):
    #     cv2.namedWindow('face' + str(i), cv2.WINDOW_NORMAL)
    #     cv2.imshow('face' + str(i), face)

    cv2.namedWindow('detect', cv2.WINDOW_NORMAL)
    cv2.imshow('detect', drawimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()