import cv2

class ssdface():
    def __init__(self, framework='caffe', threshold=0.7):
        if framework == 'caffe':
            self.net = cv2.dnn.readNetFromCaffe('ssdface/deploy.prototxt', 'ssdface/res10_300x300_ssd_iter_140000_fp16.caffemodel')
        else:
            self.net = cv2.dnn.readNetFromTensorflow('ssdface/opencv_face_detector_uint8.pb', 'ssdface/opencv_face_detector.pbtxt')
        self.conf_threshold = threshold
        self.framework = framework
    def detect(self, frame):
        frameOpencvDnn = frame.copy()
        frameHeight = frameOpencvDnn.shape[0]
        frameWidth = frameOpencvDnn.shape[1]
        if self.framework == 'caffe':
            blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], False, False)
        else:
            blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)
        self.net.setInput(blob)
        detections = self.net.forward()
        face_rois = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.conf_threshold:
                x1 = int(detections[0, 0, i, 3] * frameWidth)
                y1 = int(detections[0, 0, i, 4] * frameHeight)
                x2 = int(detections[0, 0, i, 5] * frameWidth)
                y2 = int(detections[0, 0, i, 6] * frameHeight)
                cv2.rectangle(frameOpencvDnn,(x1, y1), (x2, y2), (0, 0, 255), thickness=2)
                face_rois.append(frame[y1:y2, x1:x2])
        return frameOpencvDnn, face_rois
    def get_face(self, frame):
        frameOpencvDnn = frame.copy()
        frameHeight = frameOpencvDnn.shape[0]
        frameWidth = frameOpencvDnn.shape[1]
        if self.framework == 'caffe':
            blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], False, False)
        else:
            blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)
        self.net.setInput(blob)
        detections = self.net.forward()
        boxs, face_rois = [], []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.conf_threshold:
                x1 = int(detections[0, 0, i, 3] * frameWidth)
                y1 = int(detections[0, 0, i, 4] * frameHeight)
                x2 = int(detections[0, 0, i, 5] * frameWidth)
                y2 = int(detections[0, 0, i, 6] * frameHeight)
                boxs.append((x1, y1, x2, y2))
                face_rois.append(frame[y1:y2, x1:x2])
        return boxs, face_rois

if __name__ == "__main__" :
    ssdface_detect = ssdface(framework='caffe')
    imgpath = 's_l.jpg'
    srcimg = cv2.imread(imgpath)
    drawimg, face_rois = ssdface_detect.detect(srcimg)

    # _, face_rois = ssdface_detect.get_face(srcimg)
    # print('detect', len(face_rois), 'face')
    # for i, face in enumerate(face_rois):
    #     cv2.namedWindow('face' + str(i), cv2.WINDOW_NORMAL)
    #     cv2.imshow('face' + str(i), face)

    cv2.namedWindow('detect', cv2.WINDOW_NORMAL)
    cv2.imshow('detect', drawimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()