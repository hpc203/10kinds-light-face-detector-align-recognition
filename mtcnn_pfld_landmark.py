import numpy as np
import torch
from torchvision import transforms
import cv2
from pfld_mtcnn.pfld import PFLDInference
from pfld_mtcnn.detector import mtcnnface
from align_faces import align_process

class mtcnn_detect():
    def __init__(self, device = 'cuda', align=False):
        self.mtcnn = mtcnnface(device=device)
        self.align = align
    def detect(self, srcimg):
        bounding_boxes, landmarks = self.mtcnn.detect(srcimg)  ###landmarks: numpy array, n x 10 (x1, x2 ... x5, y1, y2 ..y5)
        if len(bounding_boxes)==0:
            return srcimg, []
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
        bounding_boxes, landmarks = self.mtcnn.detect(srcimg)
        if len(bounding_boxes)==0:
            return [], []
        boxs, face_rois = [], []
        for i in range(bounding_boxes.shape[0]):
            # score = bounding_boxes[i,4]
            box = (bounding_boxes[i, :4]).astype(np.int32).tolist()
            face_roi = srcimg[box[1]:box[3], box[0]:box[2]]
            landmark = landmarks[i, :].reshape((2, 5)).T
            if self.align:
                face_roi = align_process(srcimg, bounding_boxes[i, :4], landmark, (224,224))
            box.extend(landmark.astype(np.int32).ravel().tolist())
            boxs.append(tuple(box))
            face_rois.append(face_roi)
        return boxs, face_rois

class pfld_landmark():
    def __init__(self, device = 'cuda'):
        plfd_backbone = PFLDInference().to(device)
        plfd_backbone.load_state_dict(torch.load('pfld_mtcnn/checkpoint.pth.tar', map_location=device)['plfd_backbone'])
        self.plfd = plfd_backbone.eval()
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.device = device
    def detect(self, crop_face):
        # size = int(max(crop_face.shape[:2]) * 1.1)
        size = max(crop_face.shape[:2])
        input = cv2.resize(crop_face, (112, 112))
        input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
        input = self.transform(input).unsqueeze(0).to(self.device)
        _, landmarks = self.plfd(input)
        pre_landmark = landmarks[0]
        pre_landmark = pre_landmark.cpu().detach().numpy().reshape(-1, 2) * [size, size]
        drawimg = crop_face.copy()
        for (x, y) in pre_landmark.astype(np.int32):
            cv2.circle(drawimg, (x, y), 2, (0, 0, 255))
        return drawimg

class mtcnn_pfld():
    def __init__(self, device = 'cuda'):
        self.mtcnn = mtcnnface(device=device)
        plfd_backbone = PFLDInference().to(device)
        checkpoint = torch.load('pfld_mtcnn/checkpoint.pth.tar', map_location=device)
        plfd_backbone.load_state_dict(checkpoint['plfd_backbone'])
        self.plfd = plfd_backbone.eval()
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.device = device
    def detect(self, srcimg):
        height, width = srcimg.shape[:2]
        bounding_boxes, landmarks = self.mtcnn.detect(srcimg)
        drawimg = srcimg.copy()
        for box in bounding_boxes:
            # score = box[4]
            x1, y1, x2, y2 = (box[:4] + 0.5).astype(np.int32)
            cv2.rectangle(drawimg, (x1, y1), (x2, y2), (0, 0, 255), thickness=2)
            w = x2 - x1 + 1
            h = y2 - y1 + 1

            size = int(max([w, h]) * 1.1)
            cx = x1 + w // 2
            cy = y1 + h // 2
            x1 = cx - size // 2
            x2 = x1 + size
            y1 = cy - size // 2
            y2 = y1 + size

            dx = max(0, -x1)
            dy = max(0, -y1)
            x1 = max(0, x1)
            y1 = max(0, y1)

            edx = max(0, x2 - width)
            edy = max(0, y2 - height)
            x2 = min(width, x2)
            y2 = min(height, y2)

            cropped = srcimg[y1:y2, x1:x2]
            if (dx > 0 or dy > 0 or edx > 0 or edy > 0):
                cropped = cv2.copyMakeBorder(cropped, dy, edy, dx, edx, cv2.BORDER_CONSTANT, 0)
            input = cv2.resize(cropped, (112, 112))
            input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
            input = self.transform(input).unsqueeze(0).to(self.device)
            _, landmarks = self.plfd(input)
            pre_landmark = landmarks[0]
            pre_landmark = pre_landmark.cpu().detach().numpy().reshape(-1, 2) * [size, size]
            for (x, y) in pre_landmark.astype(np.int32):
                cv2.circle(drawimg, (x1 + x, y1 + y), 2, (0, 255, 0))
        return drawimg

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # mtcnn_face_pfld = mtcnn_pfld(device=device)
    mtcnn_face_landmark = mtcnn_detect(device=device, align=True)
    imgpath = 's_l.jpg'
    srcimg = cv2.imread(imgpath)

    # drawimg = mtcnn_face_pfld.detect(srcimg)
    drawimg, face_rois = mtcnn_face_landmark.detect(srcimg)

    # boxs, face_rois = mtcnn_face_landmark.get_face(srcimg)
    # drawimg = srcimg.copy()
    # for i,box in enumerate(boxs):
    #     cv2.rectangle(drawimg, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), thickness=2)
    #     for j in range(5):
    #         cv2.circle(drawimg, (box[4+j * 2], box[4+j * 2 + 1]), 2, (0, 255, 0), thickness=-1)

    # for i, face in enumerate(face_rois):
    #     cv2.namedWindow('face' + str(i), cv2.WINDOW_NORMAL)
    #     cv2.imshow('face' + str(i), face)

    # pfld = pfld_landmark(device=device)
    # drawimg = pfld.detect(srcimg)

    cv2.namedWindow('detect', cv2.WINDOW_NORMAL)
    cv2.imshow('detect', drawimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()