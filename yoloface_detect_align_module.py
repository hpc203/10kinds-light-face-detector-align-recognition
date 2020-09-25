from yoloface.nn.mobilenetv3 import mobilenetv3_large, mobilenetv3_large_full, mobilenetv3_small
from yoloface.nn.models import DarknetWithShh
import torch
from yoloface.utils import non_max_suppression
import cv2
import numpy as np
from yoloface.hyp import hyp
from align_faces import align_process

class yoloface():
    def __init__(self, net_type = 'mbv3_small_1_light', device = 'cuda', align=False):
        self.long_side = -1  # -1 mean origin shape
        backone = None

        assert net_type in ['mbv3_small_1', 'mbv3_small_75', 'mbv3_large_1', 'mbv3_large_75',
                            "mbv3_large_75_light", "mbv3_large_1_light", 'mbv3_small_75_light', 'mbv3_small_1_light',
                            ]

        if net_type.startswith("mbv3_small_1"):
            backone = mobilenetv3_small()
        elif net_type.startswith("mbv3_small_75"):
            backone = mobilenetv3_small(width_mult=0.75)
        elif net_type.startswith("mbv3_large_1"):
            backone = mobilenetv3_large()
        elif net_type.startswith("mbv3_large_75"):
            backone = mobilenetv3_large(width_mult=0.75)
        elif net_type.startswith("mbv3_large_f"):
            backone = mobilenetv3_large_full()

        if 'light' in net_type:
            net = DarknetWithShh(backone, hyp, light_head=True).to(device)
        else:
            net = DarknetWithShh(backone, hyp).to(device)

        self.point_num = hyp['point_num']
        weights = "yoloface/weights/{}_final.pt".format(net_type)
        net.load_state_dict(torch.load(weights, map_location=device)['model'])
        self.net = net.eval()
        self.align = align
        self.device = device
    def detect(self, srcimg):
        ori_h, ori_w, _ = srcimg.shape
        LONG_SIDE = self.long_side
        if self.long_side == -1:
            max_size = max(ori_w, ori_h)
            LONG_SIDE = max(32, max_size - max_size % 32)

        if ori_h > ori_w:
            scale_h = LONG_SIDE / ori_h
            tar_w = int(ori_w * scale_h)
            tar_w = tar_w - tar_w % 32
            tar_w = max(32, tar_w)
            tar_h = LONG_SIDE
        else:
            scale_w = LONG_SIDE / ori_w
            tar_h = int(ori_h * scale_w)
            tar_h = tar_h - tar_h % 32
            tar_h = max(32, tar_h)
            tar_w = LONG_SIDE

        scale_w = tar_w * 1.0 / ori_w
        scale_h = tar_h * 1.0 / ori_h

        image = cv2.resize(srcimg, (tar_w, tar_h))
        image = image[..., ::-1]
        image = image.astype(np.float64)
        # image = (image - hyp['mean']) / hyp['std']
        image /= 255.0
        image = np.transpose(image, [2, 0, 1])
        image = np.expand_dims(image, axis=0)
        with torch.no_grad():
            image = torch.from_numpy(image)
            image = image.to(self.device).float()
            pred = self.net(image)[0]
            pred = non_max_suppression(pred, 0.25, 0.35, multi_label=False, classes=0, agnostic=False, land=True,
                                       point_num=self.point_num)
            try:
                det = pred[0].cpu().detach().numpy()
                srcimg = srcimg.astype(np.uint8)
                det[:, :4] = det[:, :4] / np.array([scale_w, scale_h] * 2)
                det[:, 5:5 + self.point_num * 2] = det[:, 5:5 + self.point_num * 2] / np.array([scale_w, scale_h] * self.point_num)
            except:
                det = []
        drawimg, face_rois = srcimg.copy(), []
        for b in det:
            # text = "{:.4f}".format(b[4])
            b = list(map(int, b))   ###landmarks: numpy array, n x 10 (x1, y1 ... x5,y5)
            cv2.rectangle(drawimg, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), thickness=2)
            cx, cy = b[0], b[1] + 12
            # cv2.putText(drawimg, text, (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

            cv2.circle(drawimg, (b[5], b[6]), 1, (0, 0, 255), 4)
            cv2.circle(drawimg, (b[7], b[8]), 1, (0, 255, 255), 4)
            cv2.circle(drawimg, (b[9], b[10]), 1, (255, 0, 255), 4)
            cv2.circle(drawimg, (b[11], b[12]), 1, (0, 255, 0), 4)
            cv2.circle(drawimg, (b[13], b[14]), 1, (255, 0, 0), 4)
            # for i in range(5):
            #     cv2.putText(drawimg, str(i), (b[2*i+5], b[2*i+6]+12), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255))

            face_roi = srcimg[b[1]:b[3], b[0]:b[2]]
            if self.align:
                face_roi = align_process(srcimg, np.array(b[:4]), np.array(b[5:15]).reshape(-1, 2), (224,224))
            face_rois.append(face_roi)
        return drawimg, face_rois
    def get_face(self, srcimg):
        ori_h, ori_w, _ = srcimg.shape
        LONG_SIDE = self.long_side
        if self.long_side == -1:
            max_size = max(ori_w, ori_h)
            LONG_SIDE = max(32, max_size - max_size % 32)

        if ori_h > ori_w:
            scale_h = LONG_SIDE / ori_h
            tar_w = int(ori_w * scale_h)
            tar_w = tar_w - tar_w % 32
            tar_w = max(32, tar_w)
            tar_h = LONG_SIDE
        else:
            scale_w = LONG_SIDE / ori_w
            tar_h = int(ori_h * scale_w)
            tar_h = tar_h - tar_h % 32
            tar_h = max(32, tar_h)
            tar_w = LONG_SIDE

        scale_w = tar_w * 1.0 / ori_w
        scale_h = tar_h * 1.0 / ori_h

        image = cv2.resize(srcimg, (tar_w, tar_h))
        image = image[..., ::-1]
        image = image.astype(np.float64)
        # image = (image - hyp['mean']) / hyp['std']
        image /= 255.0
        image = np.transpose(image, [2, 0, 1])
        image = np.expand_dims(image, axis=0)
        with torch.no_grad():
            image = torch.from_numpy(image)
            image = image.to(self.device).float()
            pred = self.net(image)[0]
            pred = non_max_suppression(pred, 0.25, 0.35, multi_label=False, classes=0, agnostic=False, land=True,
                                       point_num=self.point_num)
            try:
                det = pred[0].cpu().detach().numpy()
                srcimg = srcimg.astype(np.uint8)
                det[:, :4] = det[:, :4] / np.array([scale_w, scale_h] * 2)
                det[:, 5:5 + self.point_num * 2] = det[:, 5:5 + self.point_num * 2] / np.array([scale_w, scale_h] * self.point_num)
            except:
                det = []
        boxs, face_rois = [], []
        for b in det:
            b = list(map(int, b))
            del b[4]  ### delte score
            boxs.append(b)

            face_roi = srcimg[b[1]:b[3], b[0]:b[2]]
            if self.align:
                face_roi = align_process(srcimg, np.array(b[:4]), np.array(b[4:14]).reshape(-1, 2), (224, 224))
            face_rois.append(face_roi)
        return boxs, face_rois
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    yoloface_detect = yoloface(device=device, align=True)
    imgpath = 's_l.jpg'
    srcimg = cv2.imread(imgpath)

    drawimg, face_rois = yoloface_detect.detect(srcimg)

    # boxs, face_rois = yoloface_detect.get_face(srcimg)
    # drawimg = srcimg.copy()
    # for i,box in enumerate(boxs):
    #     cv2.rectangle(drawimg, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), thickness=2)
    #     for j in range(5):
    #         cv2.circle(drawimg, (box[4+j * 2], box[4+j * 2 + 1]), 2, (0, 255, 0), thickness=-1)

    # for i,face in enumerate(face_rois):
    #     cv2.namedWindow('face'+str(i), cv2.WINDOW_NORMAL)
    #     cv2.imshow('face'+str(i), face)

    cv2.namedWindow('detect', cv2.WINDOW_NORMAL)
    cv2.imshow('detect', drawimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()