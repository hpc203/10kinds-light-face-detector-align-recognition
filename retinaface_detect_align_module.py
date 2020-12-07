import numpy as np
import torch
import cv2
from retinaface.detector import RetinafaceDetector, RetinafaceDetector_dnn
from align_faces import align_process

class retinaface():
    def __init__(self, device = 'cuda', align=False):
        self.retinaface = RetinafaceDetector(device=device)
        self.align = align
    def detect(self, srcimg):
        bounding_boxes, landmarks = self.retinaface.detect_faces(srcimg)
        drawimg, face_rois = srcimg.copy(), []
        for i in range(bounding_boxes.shape[0]):
            # score = bounding_boxes[i,4]
            x1, y1, x2, y2 = (bounding_boxes[i, :4]).astype(np.int32)
            cv2.rectangle(drawimg, (x1, y1), (x2, y2), (0, 0, 255), thickness=2)
            face_roi = srcimg[y1:y2, x1:x2]
            landmark = landmarks[i, :].reshape((2, 5)).T
            if self.align:
                face_roi = align_process(srcimg, bounding_boxes[i, :4], landmark, (224, 224))
            landmark = landmark.astype(np.int32)
            for j in range(5):
                cv2.circle(drawimg, (landmark[j, 0], landmark[j, 1]), 2, (0, 255, 0), thickness=-1)
                # cv2.putText(drawimg, str(j), (landmark[j, 0], landmark[j, 1] + 12), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255))
            face_rois.append(face_roi)
        return drawimg, face_rois
    def get_face(self, srcimg):
        bounding_boxes, landmarks = self.retinaface.detect_faces(srcimg)
        boxs, face_rois = [], []
        for i in range(bounding_boxes.shape[0]):
            # score = bounding_boxes[i,4]
            box = (bounding_boxes[i, :4]).astype(np.int32).tolist()
            face_roi = srcimg[box[1]:box[3], box[0]:box[2]]
            landmark = landmarks[i, :].reshape((2, 5)).T
            if self.align:
                face_roi = align_process(srcimg, bounding_boxes[i, :4], landmark, (224, 224))
            box.extend(landmark.astype(np.int32).ravel().tolist())
            boxs.append(tuple(box))
            face_rois.append(face_roi)
        return boxs, face_rois

class retinaface_dnn():
    def __init__(self, align=False):
        self.net = RetinafaceDetector_dnn()
        self.align = align

    def detect(self, srcimg):
        bounding_boxes, landmarks = self.net.detect_faces(srcimg)
        drawimg, face_rois = srcimg.copy(), []
        for i in range(bounding_boxes.shape[0]):
            # score = bounding_boxes[i,4]
            x1, y1, x2, y2 = (bounding_boxes[i, :4]).astype(np.int32)
            cv2.rectangle(drawimg, (x1, y1), (x2, y2), (0, 0, 255), thickness=2)
            face_roi = srcimg[y1:y2, x1:x2]
            landmark = landmarks[i, :].reshape((2, 5)).T
            if self.align:
                face_roi = align_process(srcimg, bounding_boxes[i, :4], landmark, (224, 224))
            landmark = landmark.astype(np.int32)
            for j in range(5):
                cv2.circle(drawimg, (landmark[j, 0], landmark[j, 1]), 2, (0, 255, 0), thickness=-1)
                # cv2.putText(drawimg, str(j), (landmark[j, 0], landmark[j, 1] + 12), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255))
            face_rois.append(face_roi)
        return drawimg, face_rois

    def get_face(self, srcimg):
        bounding_boxes, landmarks = self.net.detect_faces(srcimg)
        boxs, face_rois = [], []
        for i in range(bounding_boxes.shape[0]):
            # score = bounding_boxes[i,4]
            box = (bounding_boxes[i, :4]).astype(np.int32).tolist()
            face_roi = srcimg[box[1]:box[3], box[0]:box[2]]
            landmark = landmarks[i, :].reshape((2, 5)).T
            if self.align:
                face_roi = align_process(srcimg, bounding_boxes[i, :4], landmark, (224, 224))
            box.extend(landmark.astype(np.int32).ravel().tolist())
            boxs.append(tuple(box))
            face_rois.append(face_roi)
        return boxs, face_rois

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # retinaface_detect = retinaface(device=device, align=True)
    retinaface_detect = retinaface_dnn(align=True)
    ###dnn版本和pytorch版本的一个区别是: pytorch版本的输入图片不做resize就进入到网络里，而dnn版本的输入图片要resize到固定尺寸的,
    ###输入不同，因此对这两个版本的输出不做比较

    imgpath = 's_l.jpg'
    srcimg = cv2.imread(imgpath)

    drawimg, face_rois = retinaface_detect.detect(srcimg)

    # boxs, face_rois = retinaface_detect.get_face(srcimg)
    # drawimg = srcimg.copy()
    # for i,box in enumerate(boxs):
    #     cv2.rectangle(drawimg, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), thickness=2)
    #     for j in range(5):
    #         cv2.circle(drawimg, (box[4+j * 2], box[4+j * 2 + 1]), 2, (0, 255, 0), thickness=-1)
    #
    # for i, face in enumerate(face_rois):
    #     cv2.namedWindow('face' + str(i), cv2.WINDOW_NORMAL)
    #     cv2.imshow('face' + str(i), face)

    cv2.namedWindow('detect', cv2.WINDOW_NORMAL)
    cv2.imshow('detect', drawimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
