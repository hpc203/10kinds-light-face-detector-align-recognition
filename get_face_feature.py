import cv2
from arcface.resnet import resnet_face18
import torch
import numpy as np
import os
import pickle
import sys
from collections import OrderedDict

def convert_onnx():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = 'arcface/resnet18_110.pth'
    model = resnet_face18(use_se=False)
    # model = torch.nn.DataParallel(model)
    # model.load_state_dict(torch.load(model_path, map_location=device))

    state_dict = torch.load(model_path, map_location=device)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_state_dict[k.replace('module.', '')] = v    ## remove 'module.'
    model.load_state_dict(new_state_dict)

    model.to(device)
    model.eval()
    dummy_input = torch.randn(1, 1, 128, 128).to(device)
    onnx_path = 'arcface/resnet18_110.onnx'
    torch.onnx.export(model, dummy_input, onnx_path, input_names=['input'], output_names=['output'])

class arcface():
    def __init__(self, model_path='arcface/resnet18_110.pth', device = 'cuda'):
        self.model = resnet_face18(use_se=False)
        # self.model = torch.nn.DataParallel(self.model)
        # self.model.load_state_dict(torch.load(model_path, map_location=device))

        state_dict = torch.load(model_path, map_location=device)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k.replace('module.', '')] = v  ## remove 'module.'
        self.model.load_state_dict(new_state_dict)

        self.model.to(device)
        self.model.eval()
        self.device = device
    def get_feature(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
        img = img[np.newaxis, np.newaxis, :, :]
        # img = np.transpose(img, axes=(2,0,1))
        # img = img[np.newaxis, :, :, :]
        img = img.astype(np.float32, copy=False)
        img -= 127.5
        img /= 127.5
        with torch.no_grad():
            data = torch.from_numpy(img).to(self.device)
            output = self.model(data)
            output = output.data.cpu().numpy()
        return output

class arcface_dnn():
    def __init__(self, model_path='arcface/resnet18_110.onnx'):
        self.model = cv2.dnn.readNetFromONNX(model_path)
        self.input_size = (128, 128)
    def get_feature(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, self.input_size, interpolation=cv2.INTER_AREA)
        blob = cv2.dnn.blobFromImage(img, scalefactor=1 / 127.5, mean=127.5)
        self.model.setInput(blob)
        output = self.model.forward(['output'])
        return output

if __name__ == '__main__':
    from yoloface_detect_align_module import yoloface    ###你还可以选择其他的人脸检测器

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # face_embdnet = arcface(device=device)
    face_embdnet = arcface_dnn()   ###已调试通过，与pytorch版本的输出结果吻合
    detect_face = yoloface(device=device)

    out_emb_path = 'yoloface_detect_arcface_feature.pkl'
    imgroot = '你的文件夹绝对路径'
    dirlist = os.listdir(imgroot)    ### imgroot里有多个文件夹，每个文件夹存放着一个人物的多个肖像照，文件夹名称是人名
    feature_list, name_list = [], []
    for i,name in enumerate(dirlist):
        sys.stdout.write("\rRun person{0}, name:{1}".format(i, name))
        sys.stdout.flush()

        imgdir = os.path.join(imgroot, name)
        imglist = os.listdir(imgdir)
        for imgname in imglist:
            srcimg = cv2.imread(os.path.join(imgdir, imgname))
            _, face_img = detect_face.detect(srcimg)  ###肖像照，图片中有且仅有有一个人脸
            if len(face_img)!=1:
                continue

            feature_out = face_embdnet.get_feature(face_img[0])
            feature_list.append(np.squeeze(feature_out))
            name_list.append(name)

    if len(feature_list)>0:
        face_feature = (np.asarray(feature_list), name_list)
        with open(out_emb_path, 'wb') as f:
            pickle.dump(face_feature, f)