import torch
import cv2
import time
import numpy as np
from yoloface_detect_align_module import yoloface
from ultraface_detect_module import ultraface
from ssdface_detect_module import ssdface
from retinaface_detect_align_module import retinaface, retinaface_dnn
from mtcnn_pfld_landmark import mtcnn_detect as mtcnnface
from facebox_detect_module import facebox_pytorch as facebox
from facebox_detect_module import facebox_dnn
from dbface_detect_align_module import dbface_detect as dbface
from centerface_detect_align_module import centerface
from lffd_detect_module import lffdface
from libfacedetect_align_module import libfacedet
import matplotlib.pyplot as plt
import inspect
import argparse

def retrieve_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Object Detection using YOLO in OPENCV')
    parser.add_argument('--imgpath', type=str, default='s_l.jpg', help='Path to image file.')
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    align = False

    yoloface_detect = yoloface(device=device, align=align)
    ultraface_detect = ultraface()
    ssdface_detect = ssdface()
    retinaface_detect = retinaface(device=device, align=align)
    retinaface_dnn_detect = retinaface_dnn(align=align)
    mtcnn_detect = mtcnnface(device=device, align=align)
    facebox_detect = facebox(device=device)
    facebox_dnn_detect = facebox_dnn()
    dbface_detect = dbface(device=device, align=align)
    centerface_detect = centerface(align=align)
    lffdface_detect = lffdface(version=1)
    libface_detect = libfacedet(align=align)

    srcimg = cv2.imread(args.imgpath)
    
    a = time.time()
    yolo_result, _ = yoloface_detect.detect(srcimg)
    b = time.time()
    yolo_time = round(b - a,3)
    cv2.putText(yolo_result, 'yoloface waste time:'+str(yolo_time), (20,40), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255))

    a = time.time()
    ultraface_result, _ = ultraface_detect.detect(srcimg)
    b = time.time()
    ultraface_time = round(b - a,3)
    cv2.putText(ultraface_result, 'ultraface waste time:' + str(ultraface_time), (20, 40), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255))

    a = time.time()
    ssdface_result, _ = ssdface_detect.detect(srcimg)
    b = time.time()
    ssdface_time = round(b - a, 3)
    cv2.putText(ssdface_result, 'ssdface waste time:' + str(ssdface_time), (20, 40), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255))

    a = time.time()
    retinaface_result, _ = retinaface_detect.detect(srcimg)
    b = time.time()
    retinaface_time = round(b - a, 3)
    cv2.putText(retinaface_result, 'retinaface waste time:' + str(retinaface_time), (20, 40), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255))

    a = time.time()
    retinaface_dnn_result, _ = retinaface_dnn_detect.detect(srcimg)
    b = time.time()
    retinaface_dnn_time = round(b - a, 3)
    cv2.putText(retinaface_dnn_result, 'retinaface_dnn waste time:' + str(retinaface_dnn_time), (20, 40), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255))

    a = time.time()
    mtcnn_result, _ = mtcnn_detect.detect(srcimg)
    b = time.time()
    mtcnn_time = round(b - a, 3)
    cv2.putText(mtcnn_result, 'mtcnn waste time:' + str(mtcnn_time), (20, 40), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255))

    a = time.time()
    facebox_result, _ = facebox_detect.detect(srcimg)
    b = time.time()
    facebox_time = round(b - a, 3)
    cv2.putText(facebox_result, 'facebox waste time:' + str(facebox_time), (20, 40), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255))

    a = time.time()
    facebox_dnn_result, _ = facebox_dnn_detect.detect(srcimg)
    b = time.time()
    facebox_dnn_time = round(b - a, 3)
    cv2.putText(facebox_dnn_result, 'facebox_dnn waste time:' + str(facebox_dnn_time), (20, 40), cv2.FONT_HERSHEY_DUPLEX, 1,(0, 0, 255))

    a = time.time()
    dbface_result, _ = dbface_detect.detect(srcimg)
    b = time.time()
    dbface_time = round(b - a, 3)
    cv2.putText(dbface_result, 'dbface waste time:' + str(dbface_time), (20, 40), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255))

    a = time.time()
    centerface_result, _ = centerface_detect.detect(srcimg)
    b = time.time()
    centerface_time = round(b - a, 3)
    cv2.putText(centerface_result, 'centerface waste time:' + str(centerface_time), (20, 40), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255))

    a = time.time()
    lffdface_result, _ = lffdface_detect.detect(srcimg)
    b = time.time()
    lffdface_time = round(b - a, 3)
    cv2.putText(lffdface_result, 'lffdface waste time:' + str(lffdface_time), (20, 40), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255))

    a = time.time()
    libface_result, _ = libface_detect.detect(srcimg)
    b = time.time()
    libface_time = round(b - a, 3)
    cv2.putText(libface_result, 'libface waste time:' + str(libface_time), (20, 40), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255))

    results = (yolo_result, ultraface_result, ssdface_result, retinaface_result, retinaface_dnn_result, mtcnn_result,
               facebox_result, facebox_dnn_result, dbface_result, centerface_result, lffdface_result, libface_result)
    waste_times = (
    yolo_time, ultraface_time, ssdface_time, retinaface_time, retinaface_dnn_time, mtcnn_time, facebox_time,
    facebox_dnn_time, dbface_time, centerface_time, lffdface_time, libface_time)

    line1 = np.hstack(results[:4])
    line2 = np.hstack(results[4:8])
    line3 = np.hstack(results[8:])
    combined = np.vstack([line1, line2, line3])
    cv2.namedWindow('detect-combined', cv2.WINDOW_NORMAL)
    cv2.imshow('detect-combined', combined)
    cv2.imwrite('combined_out.jpg', combined)
    # cv2.imwrite('line1.jpg', line1)
    # cv2.imwrite('line2.jpg', line2)
    # cv2.imwrite('line3.jpg', line3)

    for i,res in enumerate(results):
        winname = retrieve_name(res)[0]
        cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
        cv2.imshow(winname, res)

    labels = []
    for data in waste_times:
        labels.append(retrieve_name(data)[0].replace('_time', ''))

    plt.rcParams['font.family'] = 'SimHei'
    x = list(range(len(waste_times)))
    plt.bar(x, waste_times, width=0.5, color='red', label='耗时比较', tick_label=labels)
    # for a, b in zip(x, waste_times):
    #     plt.text(a, b + 0.05, '%.0f' % b, ha='center', va='bottom', fontsize=10)  # 添加数据标签
    plt.xlabel("模型")
    plt.ylabel("时间")

    # plt.barh(labels, left=0, height=0.5, width=waste_times, label='耗时比较', color='red')
    # plt.ylabel("模型")
    # plt.xlabel("时间")

    plt.legend()
    plt.show()

    cv2.waitKey(0)
    cv2.destroyAllWindows()