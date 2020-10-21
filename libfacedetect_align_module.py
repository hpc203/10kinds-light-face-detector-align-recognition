import cv2
import numpy as np
from libfacedetection.priorbox import PriorBox
from libfacedetection.utils import nms
from align_faces import align_process
from pfld_mtcnn.pfld import PFLDInference
import torch
from torchvision import transforms

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
        blob = cv2.dnn.blobFromImage(img, size=self.input_shape)
        self.net.setInput(blob)
        # output_names = ['loc', 'conf']
        # loc, conf = self.net.forward(output_names)
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
        blob = cv2.dnn.blobFromImage(img, size=self.input_shape)
        self.net.setInput(blob)
        # output_names = ['loc', 'conf']
        # loc, conf = self.net.forward(output_names)
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

class libface_pfld():
    def __init__(self, conf_thresh=0.8, nms_thresh=0.3, keep_top_k=750, model_path='libfacedetection/YuFaceDetectNet_320.onnx', device = 'cuda'):
        self.net = cv2.dnn.readNet(model_path)
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.keep_top_k = keep_top_k
        inputw = int(model_path[:-5].split('_')[-1])
        inputh = int(0.75 * inputw)
        self.input_shape = (inputw, inputh)

        plfd_backbone = PFLDInference().to(device)
        checkpoint = torch.load('pfld_mtcnn/checkpoint.pth.tar', map_location=device)
        plfd_backbone.load_state_dict(checkpoint['plfd_backbone'])
        self.plfd = plfd_backbone.eval()
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.device = device
    def detect(self, img):
        h, w, _ = img.shape
        blob = cv2.dnn.blobFromImage(img, size=self.input_shape)
        self.net.setInput(blob)
        # output_names = ['loc', 'conf']
        # loc, conf = self.net.forward(output_names)
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
            drawimg = img.copy()
            for i in range(faces.shape[0]):
                # score = faces[i,-1]
                x1, y1, x2, y2 = (faces[i, :4]).astype(np.int32)
                cv2.rectangle(drawimg, (x1, y1), (x2, y2), (0, 0, 255), thickness=2)
                face_roi = img[y1:y2, x1:x2]
                # landmark = faces[i, 4:14].reshape((5, 2))
                # landmark = landmark.astype(np.int32)
                # for j in range(5):
                #     cv2.circle(drawimg, (landmark[j, 0], landmark[j, 1]), 2, (0, 255, 0), thickness=-1)
                #     # cv2.putText(drawimg, str(j), (landmark[j, 0], landmark[j, 1] + 12), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255))

                input = cv2.resize(face_roi, (112, 112))
                input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
                with torch.no_grad():
                    input = self.transform(input).unsqueeze(0).to(self.device)
                    _, landmarks = self.plfd(input)
                    pre_landmark = landmarks[0]
                    pre_landmark = pre_landmark.cpu().detach().numpy().reshape(-1, 2) * [face_roi.shape[1], face_roi.shape[0]]
                # np.save('pfld_mtcnn/pfld_pytorch_output.npy', pre_landmark)
                for (x, y) in pre_landmark.astype(np.int32):
                    cv2.circle(drawimg, (x1 + x, y1 + y), 2, (0, 255, 0), thickness=-1)
            return drawimg
        else:
            print('No faces found.')
            return img

def convert_onnx():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    plfd_backbone = PFLDInference().to(device)
    checkpoint = torch.load('pfld_mtcnn/checkpoint.pth.tar', map_location=device)
    plfd_backbone.load_state_dict(checkpoint['plfd_backbone'])
    plfd_backbone.eval()
    dummy_input = torch.randn(1, 3, 112, 112).to(device)
    onnx_path = 'pfld_mtcnn/pfld.onnx'
    torch.onnx.export(plfd_backbone, dummy_input, onnx_path, output_names=['output', 'landmarks'])
    print('convert plfd to onnx finish!!!')

class libface_pfld_dnn():
    def __init__(self, conf_thresh=0.8, nms_thresh=0.3, keep_top_k=750, model_path='libfacedetection/YuFaceDetectNet_320.onnx'):
        self.net = cv2.dnn.readNet(model_path)
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.keep_top_k = keep_top_k
        inputw = int(model_path[:-5].split('_')[-1])
        inputh = int(0.75 * inputw)
        self.input_shape = (inputw, inputh)

        self.pfld = cv2.dnn.readNetFromONNX('pfld_mtcnn/pfld.onnx')
        self.input_size = (112, 112)
    def detect(self, img):
        h, w, _ = img.shape
        blob = cv2.dnn.blobFromImage(img, size=self.input_shape)
        self.net.setInput(blob)
        # output_names = ['loc', 'conf']
        # loc, conf = self.net.forward(output_names)
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
            drawimg = img.copy()
            for i in range(faces.shape[0]):
                # score = faces[i,-1]
                x1, y1, x2, y2 = (faces[i, :4]).astype(np.int32)
                cv2.rectangle(drawimg, (x1, y1), (x2, y2), (0, 0, 255), thickness=2)
                face_roi = img[y1:y2, x1:x2]
                # landmark = faces[i, 4:14].reshape((5, 2))
                # landmark = landmark.astype(np.int32)
                # for j in range(5):
                #     cv2.circle(drawimg, (landmark[j, 0], landmark[j, 1]), 2, (0, 255, 0), thickness=-1)
                #     # cv2.putText(drawimg, str(j), (landmark[j, 0], landmark[j, 1] + 12), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255))
                blob = cv2.dnn.blobFromImage(face_roi, scalefactor=1 / 255.0, size=self.input_size, swapRB=True)
                self.pfld.setInput(blob)
                _, landmarks = self.pfld.forward(['output', 'landmarks'])
                pre_landmark = landmarks[0]
                pre_landmark = pre_landmark.reshape(-1, 2) * [face_roi.shape[1], face_roi.shape[0]]
                # np.save('pfld_mtcnn/pfld_dnn_output.npy', pre_landmark)
                for (x, y) in pre_landmark.astype(np.int32):
                    cv2.circle(drawimg, (x1 + x, y1 + y), 2, (0, 255, 0), thickness=-1)
            return drawimg
        else:
            print('No faces found.')
            return img

class pfld_dnn():
    def __init__(self, model_path='pfld_mtcnn/pfld.onnx'):
        self.net = cv2.dnn.readNetFromONNX(model_path)
        self.input_size = (112, 112)
    def detect(self, crop_img):   ###在图片中检测出的人脸区域保存成图片作为输入
        blob = cv2.dnn.blobFromImage(crop_img, scalefactor=1 / 255.0, size=self.input_size, swapRB=True)
        self.net.setInput(blob)
        _, landmarks = self.net.forward(['output', 'landmarks'])
        pre_landmark = landmarks[0]
        pre_landmark = pre_landmark.reshape(-1, 2) * [crop_img.shape[1], crop_img.shape[0]]
        drawimg = crop_img.copy()
        for (x, y) in pre_landmark.astype(np.int32):
            cv2.circle(drawimg, (x, y), 2, (0, 255, 0), thickness=-1)
        return drawimg

if __name__ == "__main__" :
    # convert_onnx()
    libface_detect = libfacedet(align=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # libface_pfld_landmark = libface_pfld(device=device)
    libface_pfld_landmark = libface_pfld_dnn()
    # pfld_landmark = pfld_dnn()

    # pfld_pytorch_output = np.load('pfld_mtcnn/pfld_pytorch_output.npy')
    # pfld_dnn_output = np.load('pfld_mtcnn/pfld_dnn_output.npy')
    # print(np.array_equal(pfld_pytorch_output, pfld_dnn_output))
    # mean_err = np.mean(pfld_pytorch_output - pfld_dnn_output)
    # print('mean_err=', mean_err)  ###误差在小数点后6位

    imgpath = 's_l.jpg'

    srcimg = cv2.imread(imgpath)
    # drawimg = pfld_landmark.detect(srcimg)
    # cv2.namedWindow('pfld_landmark', cv2.WINDOW_NORMAL)
    # cv2.imshow('pfld_landmark', drawimg)
    # cv2.waitKey(0)

    drawimg, face_rois = libface_detect.detect(srcimg)
    face_landmark = libface_pfld_landmark.detect(srcimg)
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
    cv2.namedWindow('face_landmark', cv2.WINDOW_NORMAL)
    cv2.imshow('face_landmark', face_landmark)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # imglist = ('selfie.jpg', 's_l.jpg')
    # srcimg = cv2.imread(imglist[1])
    # draw0img, _ = libface_detect.detect(srcimg)
    # cv2.namedWindow('test', cv2.WINDOW_NORMAL)
    # cv2.imshow('test', draw0img)
    # for i in range(2):
    #     srcimg = cv2.imread(imglist[i])
    #     drawimg, _ = libface_detect.detect(srcimg)
    # cv2.namedWindow('test2', cv2.WINDOW_NORMAL)
    # cv2.imshow('test2', drawimg)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()