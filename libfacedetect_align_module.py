import cv2
import numpy as np
from libfacedetection.priorbox import PriorBox
from libfacedetection.utils import nms
from align_faces import align_process

class libfacedet():
    def __init__(self, conf_thresh=0.8, nms_thresh=0.3, keep_top_k=750, model_path='libfacedetection/YuFaceDetectNet_320.onnx', align=False):
        self.net = cv2.dnn.readNet(model_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.keep_top_k = keep_top_k
        inputw = int(model_path[:-5].split('_')[-1])
        inputh = int(0.75 * inputw)
        self.input_shape = (inputw, inputh)
        self.align = align
    def detect(self, img):
        h, w, _ = img.shape
        img_resize = cv2.resize(img, dst=None, dsize=(self.input_shape), interpolation=cv2.INTER_LINEAR)
        hr, wr, _ = img_resize.shape
        blob = cv2.dnn.blobFromImage(img_resize, size=self.input_shape)
        self.net.setInput(blob)
        # output_names = ['loc', 'conf']
        # loc, conf = net.forward(output_names)
        loc, conf = self.net.forward(self.net.getUnconnectedOutLayersNames())
        # Decode bboxes and landmarks
        pb = PriorBox(input_shape=self.input_shape, output_shape=(w, h))
        dets = pb.decode(np.squeeze(loc, axis=0), np.squeeze(conf, axis=0))
        # Ignore low scores
        idx = np.where(dets[:, -1] > self.conf_thresh)[0]
        dets = dets[idx]
        # NMS
        if dets.shape[0] > 0:
            dets = nms(dets, self.nms_thresh)
            faces = dets[:self.keep_top_k, :]
            # Draw boudning boxes and landmarks on the original image
            drawimg, face_rois = img.copy(), []
            for i in range(faces.shape[0]):
                # score = faces[i,-1]
                x1, y1, x2, y2 = (faces[i, :4]).astype(np.int32)
                cv2.rectangle(drawimg, (x1, y1), (x2, y2), (0, 0, 255), thickness=2)
                face_roi = img[y1:y2, x1:x2]
                landmark = faces[i, 4:14].reshape((5, 2))
                if self.align:
                    face_roi = align_process(img, faces[i, :4], landmark, (224, 224))
                landmark = landmark.astype(np.int32)
                for j in range(5):
                    cv2.circle(drawimg, (landmark[j, 0], landmark[j, 1]), 2, (0, 255, 0), thickness=-1)
                    # cv2.putText(drawimg, str(j), (landmark[j, 0], landmark[j, 1] + 12), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255))
                face_rois.append(face_roi)
            return drawimg, face_rois
        else:
            print('No faces found.')
            return img, []
    def get_face(self, img):
        h, w, _ = img.shape
        img_resize = cv2.resize(img, dst=None, dsize=(self.input_shape), interpolation=cv2.INTER_LINEAR)
        hr, wr, _ = img_resize.shape
        blob = cv2.dnn.blobFromImage(img_resize, size=self.input_shape)
        self.net.setInput(blob)
        # output_names = ['loc', 'conf']
        # loc, conf = net.forward(output_names)
        loc, conf = self.net.forward(self.net.getUnconnectedOutLayersNames())
        # Decode bboxes and landmarks
        pb = PriorBox(input_shape=self.input_shape, output_shape=(w, h))
        dets = pb.decode(np.squeeze(loc, axis=0), np.squeeze(conf, axis=0))
        # Ignore low scores
        idx = np.where(dets[:, -1] > self.conf_thresh)[0]
        dets = dets[idx]
        # NMS
        if dets.shape[0] > 0:
            dets = nms(dets, self.nms_thresh)
            faces = dets[:self.keep_top_k, :]
            # Draw boudning boxes and landmarks on the original image
            boxs, face_rois = [], []
            for i in range(faces.shape[0]):
                # score = faces[i,-1]
                box = (faces[i, :4]).astype(np.int32).tolist()
                face_roi = img[box[1]:box[3], box[0]:box[2]]
                landmark = faces[i, 4:14].reshape((5, 2))
                if self.align:
                    face_roi = align_process(img, faces[i, :4], landmark, (224, 224))
                box.extend(landmark.astype(np.int32).ravel().tolist())
                boxs.append(tuple(box))
                face_rois.append(face_roi)
            return boxs, face_rois
        else:
            return [], []

backends = (cv2.dnn.DNN_BACKEND_DEFAULT,
            cv2.dnn.DNN_BACKEND_HALIDE,
            cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE,
            cv2.dnn.DNN_BACKEND_OPENCV)
targets = (cv2.dnn.DNN_TARGET_CPU,
           cv2.dnn.DNN_TARGET_OPENCL,
           cv2.dnn.DNN_TARGET_OPENCL_FP16,
           cv2.dnn.DNN_TARGET_MYRIAD)

if __name__ == "__main__" :
    libface_detect = libfacedet()
    imgpath = 's_l.jpg'

    srcimg = cv2.imread(imgpath)

    drawimg,face_rois = libface_detect.detect(srcimg)

    # boxs, face_rois = libface_detect.get_face(srcimg)
    # drawimg = srcimg.copy()
    # for i,box in enumerate(boxs):
    #     cv2.rectangle(drawimg, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), thickness=2)
    #     for j in range(5):
    #         cv2.circle(drawimg, (box[4+j * 2], box[4+j * 2 + 1]), 2, (0, 255, 0), thickness=-1)

    for i, face in enumerate(face_rois):
        cv2.namedWindow('face' + str(i), cv2.WINDOW_NORMAL)
        cv2.imshow('face' + str(i), face)

    cv2.namedWindow('detect', cv2.WINDOW_NORMAL)
    cv2.imshow('detect', drawimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()