import numpy as np
import torch
import cv2
from blazeface.blazeface import BlazeFace
from align_faces import align_process

class blazeface_det():
    def __init__(self, device = 'cuda', align=False):
        self.net = BlazeFace().to(device)
        self.net.load_weights("blazeface/blazeface.pth")
        self.net.load_anchors("blazeface/anchors.npy")
        self.net.min_score_thresh = 0.75
        self.net.min_suppression_threshold = 0.3
        self.align = align
    def detect(self, srcimg):
        img = cv2.cvtColor(cv2.resize(srcimg, (128,128)), cv2.COLOR_BGR2RGB)
        detections = self.net.predict_on_image(img)
        if isinstance(detections, torch.Tensor):
            detections = detections.cpu().numpy()
        if detections.ndim == 1:
            detections = np.expand_dims(detections, axis=0)
        drawimg, face_rois = srcimg.copy(), []
        for i in range(detections.shape[0]):
            ymin = int(detections[i, 0] * img.shape[0])
            xmin = int(detections[i, 1] * img.shape[1])
            ymax = int(detections[i, 2] * img.shape[0])
            xmax = int(detections[i, 3] * img.shape[1])
            cv2.rectangle(drawimg, (xmin, ymin), (xmax, ymax), (0, 0, 255), thickness=2)
            face_roi = srcimg[ymin:ymax, xmin:xmax]
            for k in range(6):
                kp_x = int(detections[i, 4 + k * 2] * img.shape[1])
                kp_y = int(detections[i, 4 + k * 2 + 1] * img.shape[0])
                cv2.circle(drawimg, (kp_x, kp_y), 1, (0, 255, 0), 1)
                cv2.putText(drawimg, str(k), (kp_x, kp_y + 12), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 0, 0))

            face_rois.append(face_roi)
        return drawimg, face_rois

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    blazeface_detect = blazeface_det(device=device, align=True)
    imgpath = 's_l.jpg'
    srcimg = cv2.imread(imgpath)

    drawimg, face_rois = blazeface_detect.detect(srcimg)

    # boxs, face_rois = blazeface_detect.get_face(srcimg)
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