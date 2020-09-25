import cv2
from arcface.resnet import resnet_face18
import torch
import numpy as np
import os
import pickle
import sys

class arcface():
    def __init__(self, model_path='arcface/resnet18_110.pth', device = 'cuda'):
        self.model = resnet_face18(use_se=False)
        self.model = torch.nn.DataParallel(self.model)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
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

if __name__ == '__main__':
    from yoloface_detect_align_module import yoloface    ###你还可以选择其他的人脸检测器

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    face_embdnet = arcface(device=device)
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
            feature_list.append(feature_out)
            name_list.append(name)

    face_feature = (np.squeeze(np.asarray(feature_list)), name_list)
    with open(out_emb_path, 'wb') as f:
        pickle.dump(face_feature, f)