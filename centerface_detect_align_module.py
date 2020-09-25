import numpy as np
import cv2
from align_faces import align_process

class centerface():
    def __init__(self, landmarks=True, align=False):
        self.landmarks = landmarks
        # if self.landmarks:
        #     # self.net = cv2.dnn.readNetFromONNX('centerface/centerface.onnx')
        #     self.net = cv2.dnn.readNet('centerface/centerface.onnx')
        # else:
        #     # self.net = cv2.dnn.readNetFromONNX('centerface/cface.1k.onnx')
        #     self.net = cv2.dnn.readNet('centerface/cface.1k.onnx')
        self.img_h_new, self.img_w_new, self.scale_h, self.scale_w = 0, 0, 0, 0
        self.align = align
    def inference_opencv(self, img, threshold):
        if self.landmarks:    ###restart dnn
            self.net = cv2.dnn.readNet('centerface/centerface.onnx')
        else:
            self.net = cv2.dnn.readNet('centerface/cface.1k.onnx')
        blob = cv2.dnn.blobFromImage(img, scalefactor=1.0, size=(self.img_w_new, self.img_h_new), mean=(0, 0, 0), swapRB=True, crop=False)
        self.net.setInput(blob)
        if self.landmarks:
            heatmap, scale, offset, lms = self.net.forward(["537", "538", "539", '540'])
        else:
            heatmap, scale, offset = self.net.forward(["535", "536", "537"])
            lms = np.array([])
        return self.postprocess(heatmap, lms, offset, scale, threshold)

    def transform(self, h, w):
        img_h_new, img_w_new = int(np.ceil(h / 32) * 32), int(np.ceil(w / 32) * 32)
        scale_h, scale_w = img_h_new / h, img_w_new / w
        return img_h_new, img_w_new, scale_h, scale_w

    def postprocess(self, heatmap, lms, offset, scale, threshold):
        dets, lms = self.decode(heatmap, scale, offset, lms, (self.img_h_new, self.img_w_new), threshold=threshold)
        if len(dets) > 0:
            dets[:, 0:4:2], dets[:, 1:4:2] = dets[:, 0:4:2] / self.scale_w, dets[:, 1:4:2] / self.scale_h
            if self.landmarks:
                lms[:, 0:10:2], lms[:, 1:10:2] = lms[:, 0:10:2] / self.scale_w, lms[:, 1:10:2] / self.scale_h
        else:
            dets = np.empty(shape=[0, 5], dtype=np.float32)
            if self.landmarks:
                lms = np.empty(shape=[0, 10], dtype=np.float32)
        return dets, lms

    def decode(self, heatmap, scale, offset, landmark, size, threshold=0.1):
        heatmap = np.squeeze(heatmap)
        scale0, scale1 = scale[0, 0, :, :], scale[0, 1, :, :]
        offset0, offset1 = offset[0, 0, :, :], offset[0, 1, :, :]
        c0, c1 = np.where(heatmap > threshold)
        boxes, lms = [], []
        if len(c0) > 0:
            for i in range(len(c0)):
                s0, s1 = np.exp(scale0[c0[i], c1[i]]) * 4, np.exp(scale1[c0[i], c1[i]]) * 4
                o0, o1 = offset0[c0[i], c1[i]], offset1[c0[i], c1[i]]
                s = heatmap[c0[i], c1[i]]
                x1, y1 = max(0, (c1[i] + o1 + 0.5) * 4 - s1 / 2), max(0, (c0[i] + o0 + 0.5) * 4 - s0 / 2)
                x1, y1 = min(x1, size[1]), min(y1, size[0])
                boxes.append([x1, y1, min(x1 + s1, size[1]), min(y1 + s0, size[0]), s])
                if self.landmarks:
                    lm = []
                    for j in range(5):
                        lm.append(landmark[0, j * 2 + 1, c0[i], c1[i]] * s1 + x1)
                        lm.append(landmark[0, j * 2, c0[i], c1[i]] * s0 + y1)
                    lms.append(lm)
            boxes = np.asarray(boxes, dtype=np.float32)
            keep = self.nms(boxes[:, :4], boxes[:, 4], 0.3)
            boxes = boxes[keep, :]
            lms = np.asarray(lms, dtype=np.float32)
            if self.landmarks:
                lms = lms[keep, :]
        return boxes, lms

    def nms(self, boxes, scores, nms_thresh):
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = np.argsort(scores)[::-1]
        num_detections = boxes.shape[0]
        suppressed = np.zeros((num_detections,), dtype=np.bool)

        keep = []
        for _i in range(num_detections):
            i = order[_i]
            if suppressed[i]:
                continue
            keep.append(i)

            ix1 = x1[i]
            iy1 = y1[i]
            ix2 = x2[i]
            iy2 = y2[i]
            iarea = areas[i]

            for _j in range(_i + 1, num_detections):
                j = order[_j]
                if suppressed[j]:
                    continue

                xx1 = max(ix1, x1[j])
                yy1 = max(iy1, y1[j])
                xx2 = min(ix2, x2[j])
                yy2 = min(iy2, y2[j])
                w = max(0, xx2 - xx1 + 1)
                h = max(0, yy2 - yy1 + 1)

                inter = w * h
                ovr = inter / (iarea + areas[j] - inter)
                if ovr >= nms_thresh:
                    suppressed[j] = True

        return keep

    def detect(self, srcimg, threshold=0.5):
        height, width = srcimg.shape[:2]
        self.img_h_new, self.img_w_new, self.scale_h, self.scale_w = self.transform(height, width)
        dets, lms = self.inference_opencv(srcimg, threshold)

        drawimg, face_rois = srcimg.copy(), []
        for i in range(dets.shape[0]):
            boxes, score = dets[i, :4], dets[i, 4]
            cv2.rectangle(drawimg, (int(boxes[0]), int(boxes[1])), (int(boxes[2]), int(boxes[3])), (0, 0, 255), thickness=2)
            face_roi = srcimg[int(boxes[1]):int(boxes[3]), int(boxes[0]):int(boxes[2])]

            if len(lms) > 0:
                lm = lms[i, :]   ###landmarks: numpy array, n x 10 (x1, y1 ... x5,y5)
                for j in range(0,5):
                    cv2.circle(drawimg, (int(lm[j * 2]), int(lm[j * 2 + 1])), 2, (0, 255, 0), thickness=-1)
                    # cv2.putText(drawimg, str(j), (int(lm[j * 2]), int(lm[j * 2 + 1]) + 12), cv2.FONT_HERSHEY_DUPLEX, 1,(0, 0, 255))
                if self.align:
                    face_roi = align_process(srcimg, np.array(boxes), np.array(lm).reshape(-1, 2), (224, 224))
            face_rois.append(face_roi)
        return drawimg, face_rois
    def get_face(self, srcimg, threshold=0.5):
        height, width = srcimg.shape[:2]
        self.img_h_new, self.img_w_new, self.scale_h, self.scale_w = self.transform(height, width)
        dets, lms = self.inference_opencv(srcimg, threshold)
        boxs, face_rois = [], []
        for i in range(dets.shape[0]):
            # boxes, score = dets[i, :4], dets[i, 4]
            box = [int(dets[i, 0]), int(dets[i, 1]), int(dets[i, 2]), int(dets[i, 3])]
            face_roi = srcimg[box[1]:box[3], box[0]:box[2]]

            if len(lms) > 0:
                lm = lms[i, :]
                if self.align:
                    face_roi = align_process(srcimg, np.array(box), np.array(lm).reshape(-1, 2), (224, 224))
                box.extend(list(map(int, lm.tolist())))
            boxs.append(tuple(box))
            face_rois.append(face_roi)
        return boxs, face_rois

if __name__ == "__main__":
    centerface_detect = centerface()
    imgpath = 's_l.jpg'
    srcimg = cv2.imread(imgpath)
    drawimg, face_rois = centerface_detect.detect(srcimg)

    # boxs, face_rois = centerface_detect.get_face(srcimg)
    # drawimg = srcimg.copy()
    # for i, box in enumerate(boxs):
    #     cv2.rectangle(drawimg, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), thickness=2)
    #     for j in range(5):
    #         cv2.circle(drawimg, (box[4 + j * 2], box[4 + j * 2 + 1]), 2, (0, 255, 0), thickness=-1)

    # print('detect',len(face_rois),'face')
    # for i, face in enumerate(face_rois):
    #     cv2.namedWindow('face' + str(i), cv2.WINDOW_NORMAL)
    #     cv2.imshow('face' + str(i), face)

    cv2.namedWindow('detect', cv2.WINDOW_NORMAL)
    cv2.imshow('detect', drawimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()