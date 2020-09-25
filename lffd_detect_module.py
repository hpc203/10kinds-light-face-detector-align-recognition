import cv2
import numpy as np

class lffdface():
    def __init__(self, version=2):
        if version == 2:
            self.net = cv2.dnn.readNetFromCaffe('lffdface/symbol_10_320_20L_5scales_v2_deploy.prototxt', 'lffdface/train_10_320_20L_5scales_v2_iter_1000000.caffemodel')
            self.receptive_field_list = (20, 40, 80, 160, 320)
            self.receptive_field_stride = (4, 8, 16, 32, 64)
            self.receptive_field_center_start = (3, 7, 15, 31, 63)
            self.num_output_scales = 5
            self.constant = [i / 2.0 for i in self.receptive_field_list]
        else:
            self.net = cv2.dnn.readNetFromCaffe('lffdface/symbol_10_560_25L_8scales_v1_deploy.prototxt', 'lffdface/train_10_560_25L_8scales_v1_iter_1400000.caffemodel')
            self.receptive_field_list = (15, 20, 40, 70, 110, 250, 400, 560)
            self.receptive_field_stride = (4, 4, 8, 8, 16, 32, 32, 32)
            self.receptive_field_center_start = (3, 3, 7, 7, 15, 31, 31, 31)
            self.num_output_scales = 8
            self.constant = [i / 2.0 for i in self.receptive_field_list]

    def getOutputsNames(self):
        # Get the names of all the layers in the network
        layersNames = self.net.getLayerNames()
        # print(dir(net))
        # Get the names of the output layers, i.e. the layers with unconnected outputs
        return [layersNames[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

    def NMS(self, boxes, overlap_threshold):
        '''

        :param boxes: np nx5, n is the number of boxes, 0:4->x1, y1, x2, y2, 4->score
        :param overlap_threshold:
        :return:
        '''
        if boxes.shape[0] == 0:
            return boxes

        # if the bounding boxes integers, convert them to floats --
        # this is important since we'll be doing a bunch of divisions
        if boxes.dtype != np.float32:
            boxes = boxes.astype(np.float32)

        # initialize the list of picked indexes
        pick = []
        # grab the coordinates of the bounding boxes
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        sc = boxes[:, 4]
        widths = x2 - x1
        heights = y2 - y1

        # compute the area of the bounding boxes and sort the bounding
        # boxes by the bottom-right y-coordinate of the bounding box
        area = heights * widths
        idxs = np.argsort(sc)  # 从小到大排序

        # keep looping while some indexes still remain in the indexes list
        while len(idxs) > 0:
            # grab the last index in the indexes list and add the
            # index value to the list of picked indexes
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            # compare secend highest score boxes
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            # compute the width and height of the bo（ box
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            # compute the ratio of overlap
            overlap = (w * h) / area[idxs[:last]]

            # delete all indexes from the index list that have
            idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_threshold)[0])))

        # return only the bounding boxes that were picked using the
        # integer data type
        return boxes[pick]
    def detect(self, image, resize_scale=1, score_threshold=0.3, top_k=10000, NMS_threshold=0.3, NMS_flag=True, skip_scale_branch_list=[]):
        shorter_side = min(image.shape[:2])
        if shorter_side * resize_scale < 128:
            resize_scale = float(128) / shorter_side

        input_image = cv2.resize(image, (0, 0), fx=resize_scale, fy=resize_scale)
        blob = cv2.dnn.blobFromImage(input_image, scalefactor=1/127.5, size=None, mean=[127.5, 127.5, 127.5])

        self.net.setInput(blob)
        # outputs = self.net.forward(self.getOutputsNames())
        outputs = self.net.forward(self.net.getUnconnectedOutLayersNames())
        bbox_collection = []
        for i in range(self.num_output_scales):
            if i in skip_scale_branch_list:
                continue

            score_map = np.squeeze(outputs[i * 2], (0, 1))
            bbox_map = np.squeeze(outputs[i * 2 + 1], 0)

            RF_center_Xs = np.array(
                [self.receptive_field_center_start[i] + self.receptive_field_stride[i] * x for x in
                 range(score_map.shape[1])])
            RF_center_Xs_mat = np.tile(RF_center_Xs, [score_map.shape[0], 1])
            RF_center_Ys = np.array(
                [self.receptive_field_center_start[i] + self.receptive_field_stride[i] * y for y in
                 range(score_map.shape[0])])
            RF_center_Ys_mat = np.tile(RF_center_Ys, [score_map.shape[1], 1]).T

            x_lt_mat = RF_center_Xs_mat - bbox_map[0, :, :] * self.constant[i]
            y_lt_mat = RF_center_Ys_mat - bbox_map[1, :, :] * self.constant[i]
            x_rb_mat = RF_center_Xs_mat - bbox_map[2, :, :] * self.constant[i]
            y_rb_mat = RF_center_Ys_mat - bbox_map[3, :, :] * self.constant[i]

            x_lt_mat = x_lt_mat / resize_scale
            x_lt_mat[x_lt_mat < 0] = 0
            y_lt_mat = y_lt_mat / resize_scale
            y_lt_mat[y_lt_mat < 0] = 0
            x_rb_mat = x_rb_mat / resize_scale
            x_rb_mat[x_rb_mat > image.shape[1]] = image.shape[1]
            y_rb_mat = y_rb_mat / resize_scale
            y_rb_mat[y_rb_mat > image.shape[0]] = image.shape[0]

            select_index = np.where(score_map > score_threshold)
            for idx in range(select_index[0].size):
                bbox_collection.append((x_lt_mat[select_index[0][idx], select_index[1][idx]],
                                        y_lt_mat[select_index[0][idx], select_index[1][idx]],
                                        x_rb_mat[select_index[0][idx], select_index[1][idx]],
                                        y_rb_mat[select_index[0][idx], select_index[1][idx]],
                                        score_map[select_index[0][idx], select_index[1][idx]]))

            # NMS
        bbox_collection = sorted(bbox_collection, key=lambda item: item[-1], reverse=True)
        if len(bbox_collection) > top_k:
            bbox_collection = bbox_collection[0:top_k]
        bbox_collection_np = np.array(bbox_collection, dtype=np.float32)

        if NMS_flag:
            final_bboxes = self.NMS(bbox_collection_np, NMS_threshold)
            face_rois, drawimg = [], image.copy()
            for i in range(final_bboxes.shape[0]):
                xmin, ymin, xmax, ymax = final_bboxes[i,:4].astype(np.int)
                cv2.rectangle(drawimg, (xmin, ymin), (xmax, ymax), (0, 0, 255), thickness=2)
                face_rois.append(image[ymin:ymax, xmin:xmax])
            return drawimg, face_rois
        else:
            face_rois, drawimg = [], image.copy()
            for i in range(bbox_collection_np.shape[0]):
                xmin, ymin, xmax, ymax = bbox_collection_np[i,:4].astype(np.int)
                cv2.rectangle(drawimg, (xmin, ymin), (xmax, ymax), (0, 0, 255), thickness=2)
                face_rois.append(image[ymin:ymax, xmin:xmax])
            return drawimg, face_rois
    def get_face(self, image, resize_scale=1, score_threshold=0.3, top_k=10000, NMS_threshold=0.3, NMS_flag=True, skip_scale_branch_list=[]):
        shorter_side = min(image.shape[:2])
        if shorter_side * resize_scale < 128:
            resize_scale = float(128) / shorter_side

        input_image = cv2.resize(image, (0, 0), fx=resize_scale, fy=resize_scale)
        blob = cv2.dnn.blobFromImage(input_image, scalefactor=1 / 127.5, size=None, mean=[127.5, 127.5, 127.5])

        self.net.setInput(blob)
        # outputs = self.net.forward(self.getOutputsNames())
        outputs = self.net.forward(self.net.getUnconnectedOutLayersNames())
        bbox_collection = []
        for i in range(self.num_output_scales):
            if i in skip_scale_branch_list:
                continue

            score_map = np.squeeze(outputs[i * 2], (0, 1))
            bbox_map = np.squeeze(outputs[i * 2 + 1], 0)

            RF_center_Xs = np.array(
                [self.receptive_field_center_start[i] + self.receptive_field_stride[i] * x for x in
                 range(score_map.shape[1])])
            RF_center_Xs_mat = np.tile(RF_center_Xs, [score_map.shape[0], 1])
            RF_center_Ys = np.array(
                [self.receptive_field_center_start[i] + self.receptive_field_stride[i] * y for y in
                 range(score_map.shape[0])])
            RF_center_Ys_mat = np.tile(RF_center_Ys, [score_map.shape[1], 1]).T

            x_lt_mat = RF_center_Xs_mat - bbox_map[0, :, :] * self.constant[i]
            y_lt_mat = RF_center_Ys_mat - bbox_map[1, :, :] * self.constant[i]
            x_rb_mat = RF_center_Xs_mat - bbox_map[2, :, :] * self.constant[i]
            y_rb_mat = RF_center_Ys_mat - bbox_map[3, :, :] * self.constant[i]

            x_lt_mat = x_lt_mat / resize_scale
            x_lt_mat[x_lt_mat < 0] = 0
            y_lt_mat = y_lt_mat / resize_scale
            y_lt_mat[y_lt_mat < 0] = 0
            x_rb_mat = x_rb_mat / resize_scale
            x_rb_mat[x_rb_mat > image.shape[1]] = image.shape[1]
            y_rb_mat = y_rb_mat / resize_scale
            y_rb_mat[y_rb_mat > image.shape[0]] = image.shape[0]

            select_index = np.where(score_map > score_threshold)
            for idx in range(select_index[0].size):
                bbox_collection.append((x_lt_mat[select_index[0][idx], select_index[1][idx]],
                                        y_lt_mat[select_index[0][idx], select_index[1][idx]],
                                        x_rb_mat[select_index[0][idx], select_index[1][idx]],
                                        y_rb_mat[select_index[0][idx], select_index[1][idx]],
                                        score_map[select_index[0][idx], select_index[1][idx]]))

            # NMS
        bbox_collection = sorted(bbox_collection, key=lambda item: item[-1], reverse=True)
        if len(bbox_collection) > top_k:
            bbox_collection = bbox_collection[0:top_k]
        bbox_collection_np = np.array(bbox_collection, dtype=np.float32)

        if NMS_flag:
            final_bboxes = self.NMS(bbox_collection_np, NMS_threshold)
            boxs, face_rois = [], []
            for i in range(final_bboxes.shape[0]):
                xmin, ymin, xmax, ymax = final_bboxes[i, :4].astype(np.int)
                boxs.append((xmin, ymin, xmax, ymax))
                face_rois.append(image[ymin:ymax, xmin:xmax])
            return boxs, face_rois
        else:
            boxs, face_rois = [], []
            for i in range(bbox_collection_np.shape[0]):
                xmin, ymin, xmax, ymax = bbox_collection_np[i, :4].astype(np.int)
                boxs.append((xmin, ymin, xmax, ymax))
                face_rois.append(image[ymin:ymax, xmin:xmax])
            return boxs, face_rois

if __name__ == "__main__" :
    lffdface_detect = lffdface(version=2)
    imgpath = 's_l.jpg'

    srcimg = cv2.imread(imgpath)
    drawimg, face_rois = lffdface_detect.detect(srcimg)

    # _, face_rois = lffdface_detect.get_face(srcimg)
    # for i, face in enumerate(face_rois):
    #     cv2.namedWindow('face' + str(i), cv2.WINDOW_NORMAL)
    #     cv2.imshow('face' + str(i), face)

    cv2.namedWindow('detect', cv2.WINDOW_NORMAL)
    cv2.imshow('detect', drawimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()