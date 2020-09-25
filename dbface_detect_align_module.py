import dbface.common as common
import torch
import numpy as np
import cv2
from dbface.DBFaceSmallH import DBFace as dbface_small
from dbface.DBFace import DBFace as dbface
import torch.nn.functional as F
from align_faces import align_process

class dbface_detect():
    def __init__(self, net_type = 'dbface_small', device = 'cuda', align=False):
        if net_type == 'dbface_small':
            self.net = dbface_small().to(device)
            self.net.load_state_dict(torch.load('dbface/dbfaceSmallH.pth', map_location=device))
            self.net.eval()
        else:
            self.net = dbface().to(device)
            self.net.load_state_dict(torch.load('dbface/dbface.pth', map_location=device))
            self.net.eval()
        self.align = align
        self.device = device
        self.mean = np.array([0.408, 0.447, 0.47], dtype=np.float32).reshape((1, 1, -1))   ###广播法则
        self.std = np.array([0.289, 0.274, 0.278], dtype=np.float32).reshape((1, 1, -1))
    def nms(self, objs, iou=0.5):
        if objs is None or len(objs) <= 1:
            return objs
        objs = sorted(objs, key=lambda obj: obj.score, reverse=True)
        keep = []
        flags = [0] * len(objs)
        for index, obj in enumerate(objs):

            if flags[index] != 0:
                continue

            keep.append(obj)
            for j in range(index + 1, len(objs)):
                if flags[j] == 0 and obj.iou(objs[j]) > iou:
                    flags[j] = 1
        return keep
    def detect(self, srcimg, threshold=0.3, nms_iou=0.3):
        image = common.pad(srcimg).astype(np.float32)
        image = ((image / 255.0 - self.mean) / self.std)
        image = image.transpose(2, 0, 1)
        with torch.no_grad():
            torch_image = torch.from_numpy(image)[None]
            torch_image = torch_image.to(self.device)

            hm, box, landmark = self.net(torch_image)
            hm_pool = F.max_pool2d(hm, 3, 1, 1)
            scores, indices = ((hm == hm_pool).float() * hm).view(1, -1).cpu().topk(1000)
            hm_height, hm_width = hm.shape[2:]

            scores = scores.squeeze()
            indices = indices.squeeze()
            ys = list((indices / hm_width).int().data.numpy())
            xs = list((indices % hm_width).int().data.numpy())
            scores = list(scores.data.numpy())
            box = box.cpu().squeeze().data.numpy()
            landmark = landmark.cpu().squeeze().data.numpy()

            stride = 4
            objs = []
            for cx, cy, score in zip(xs, ys, scores):
                if score < threshold:
                    break

                x, y, r, b = box[:, cy, cx]
                xyrb = (np.array([cx, cy, cx, cy]) + [-x, -y, r, b]) * stride
                x5y5 = landmark[:, cy, cx]
                x5y5 = (common.exp(x5y5 * 4) + ([cx] * 5 + [cy] * 5)) * stride
                box_landmark = list(zip(x5y5[:5], x5y5[5:]))
                objs.append(common.BBox(0, xyrb=xyrb, score=score, landmark=box_landmark))
            objs = self.nms(objs, iou=nms_iou)
        drawimg, face_rois = srcimg.copy(), []
        for i, obj in enumerate(objs):
            box, score, landmark = list(map(int, obj.box)), obj.score, obj.landmark
            landmark = [int(x) for t in obj.landmark for x in t]
            # landmark = sum(obj.landmark, ())
            # landmark = list(itertools.chain.from_iterable(obj.landmark))
            cv2.rectangle(drawimg, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), thickness=2)
            for j in range(0, 5):
                cv2.circle(drawimg, (landmark[j * 2], landmark[j * 2 + 1]), 2, (0, 255, 0), thickness=-1)
                # cv2.putText(drawimg, str(j), (landmark[j * 2], landmark[j * 2 + 1] + 12), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255))
            face_roi = srcimg[box[1]:box[3], box[0]:box[2]]
            # import imutils
            # angle = np.rad2deg(np.arctan2(landmark[3] - landmark[1], landmark[2] - landmark[0]))
            # if angle != 0 and self.align:
            #     face_roi = imutils.rotate(face_roi, angle)
            if self.align:
                face_roi = align_process(srcimg, np.array(box), np.array(landmark).reshape(-1, 2), (224,224))
            face_rois.append(face_roi)
        return drawimg, face_rois
    def get_face(self, srcimg, threshold=0.3, nms_iou=0.3):
        image = common.pad(srcimg).astype(np.float32)
        image = ((image / 255.0 - self.mean) / self.std)
        image = image.transpose(2, 0, 1)
        with torch.no_grad():
            torch_image = torch.from_numpy(image)[None]
            torch_image = torch_image.to(self.device)

            hm, box, landmark = self.net(torch_image)
            hm_pool = F.max_pool2d(hm, 3, 1, 1)
            scores, indices = ((hm == hm_pool).float() * hm).view(1, -1).cpu().topk(1000)
            hm_height, hm_width = hm.shape[2:]

            scores = scores.squeeze()
            indices = indices.squeeze()
            ys = list((indices / hm_width).int().data.numpy())
            xs = list((indices % hm_width).int().data.numpy())
            scores = list(scores.data.numpy())
            box = box.cpu().squeeze().data.numpy()
            landmark = landmark.cpu().squeeze().data.numpy()

            stride = 4
            objs = []
            for cx, cy, score in zip(xs, ys, scores):
                if score < threshold:
                    break

                x, y, r, b = box[:, cy, cx]
                xyrb = (np.array([cx, cy, cx, cy]) + [-x, -y, r, b]) * stride
                x5y5 = landmark[:, cy, cx]
                x5y5 = (common.exp(x5y5 * 4) + ([cx] * 5 + [cy] * 5)) * stride
                box_landmark = list(zip(x5y5[:5], x5y5[5:]))
                objs.append(common.BBox(0, xyrb=xyrb, score=score, landmark=box_landmark))
            objs = self.nms(objs, iou=nms_iou)
        boxs, face_rois = [], []
        for i, obj in enumerate(objs):
            box, score, landmark = list(map(int, obj.box)), obj.score, obj.landmark
            landmark = [int(x) for t in obj.landmark for x in t]
            boxs.append(box+landmark)

            face_roi = srcimg[box[1]:box[3], box[0]:box[2]]
            if self.align:
                face_roi = align_process(srcimg, np.array(box), np.array(landmark).reshape(-1, 2), (224, 224))
            face_rois.append(face_roi)
        return boxs, face_rois

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    DBface_Detect = dbface_detect(device=device, align=True)
    imgpath = 's_l.jpg'
    srcimg = cv2.imread(imgpath)

    drawimg, face_rois = DBface_Detect.detect(srcimg)

    # boxs, face_rois = dbface_detect.get_face(srcimg)
    # drawimg = srcimg.copy()
    # for i, box in enumerate(boxs):
    #     cv2.rectangle(drawimg, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), thickness=2)
    #     for j in range(5):
    #         cv2.circle(drawimg, (box[4 + j * 2], box[4 + j * 2 + 1]), 2, (0, 255, 0), thickness=-1)

    # for i, face in enumerate(face_rois):
    #     cv2.namedWindow('face' + str(i), cv2.WINDOW_NORMAL)
    #     cv2.imshow('face' + str(i), face)

    cv2.namedWindow('detect', cv2.WINDOW_NORMAL)
    cv2.imshow('detect', drawimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()