import torch
from yoloface_detect_align_module import yoloface
from get_face_feature import arcface
import pickle
import cv2
import numpy as np
from scipy import spatial
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Object Detection using YOLO in OPENCV')
    parser.add_argument('--imgpath', type=str, default='s_l.jpg', help='Path to image file.')
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    face_embdnet = arcface(device=device)
    detect_face = yoloface(device=device)
    emb_path = 'yolo_detect_arcface_feature.pkl'
    with open(emb_path, 'rb') as f:
        dataset = pickle.load(f)
    faces_feature, names_list = dataset

    srcimg = cv2.imread(args.imgpath)
    if srcimg is None:
        exit('please give correct image')
    boxs, faces_img = detect_face.get_face(srcimg)
    if len(faces_img) == 0:
        exit('no detec face')
    drawimg, threshold = srcimg.copy(), 0.65
    for i, face in enumerate(faces_img):
        feature_out = face_embdnet.get_feature(face)
        dist = spatial.distance.cdist(faces_feature, feature_out, metric='euclidean').flatten()
        min_id = np.argmin(dist)
        pred_score = dist[min_id]
        pred_name = 'unknow'
        if dist[min_id] <= threshold:
            pred_name = names_list[min_id]
        cv2.rectangle(drawimg, (boxs[i][0], boxs[i][1]), (boxs[i][2], boxs[i][3]), (0, 0, 255), thickness=2)
        cv2.putText(drawimg, pred_name, (boxs[i][0], boxs[i][1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
    cv2.namedWindow('face recognition', cv2.WINDOW_NORMAL)
    cv2.imshow('face recognition', drawimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()