import cv2
import numpy as np

# reference facial points, a list of coordinates (x,y)
REFERENCE_FACIAL_POINTS = [
    [30.29459953, 51.69630051],
    [65.53179932, 51.50139999],
    [48.02519989, 71.73660278],
    [33.54930115, 92.3655014],
    [62.72990036, 92.20410156]
]

# def face_alignment(im, kpts_locations, kpts_locations_ref, FACE_SIZE_REF=(200,200)):
#     M = cv2.estimateAffine2D(kpts_locations, kpts_locations_ref)
#     face_im_aligned = cv2.warpAffine(im, M[0], FACE_SIZE_REF)
#     return face_im_aligned

def align_process(img, bbox, landmark, image_size):
    """
    crop and align face
    Parameters:
    ----------
        img: numpy array, bgr order of shape (1, 3, n, m)
            input image
        points: numpy array, n x 10 (x1, x2 ... x5, y1, y2 ..y5)
        desired_size: default 256
        padding: default 0
    Retures:
    -------
    crop_imgs: list, n
        cropped and aligned faces
    """
    M = None
    # image_size = []
    # str_image_size = kwargs.get('image_size', '')
    # if len(str_image_size) > 0:
    #     image_size = [int(x) for x in str_image_size.split(',')]
    #     if len(image_size) == 1:
    #         image_size = [image_size[0], image_size[0]]
    #     assert len(image_size) == 2
    #     assert image_size[0] == image_size[1]
    #     assert image_size[0] % 2 == 0
    if landmark is not None:
        assert len(image_size) == 2
        # 这个基准是112*96的面部特征点的坐标
        src = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041]], dtype=np.float32)

        # if image_size[0] != 112:
        #     src[:, 1] += (image_size[0] - 112)/2
        # if image_size[1] != 96:
        #     src[:, 0] += (image_size[1] - 96)/2
        # if image_size[1] != 112:
        src[:, 0] += 8
        # make the mark points up 8 pixels, for crop the chin in the cropped image
        src[:, 1] -= 8

        if image_size[0] == image_size[1] and image_size[0] != 112:
            src = src / 112 * image_size[0]

        dst = landmark.astype(np.float32)
        M, _ = cv2.estimateAffinePartial2D(dst.reshape(1,5,2), src.reshape(1,5,2))

    if M is None:
        if bbox is None:  # use center crop
            det = np.zeros(4, dtype=np.int32)
            det[0] = int(img.shape[1] * 0.0625)
            det[1] = int(img.shape[0] * 0.0625)
            det[2] = img.shape[1] - det[0]
            det[3] = img.shape[0] - det[1]
        else:
            det = bbox
        # margin = kwargs.get('margin', 44)
        margin = 44
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0] - margin / 2, 0)
        bb[1] = np.maximum(det[1] - margin / 2, 0)
        bb[2] = np.minimum(det[2] + margin / 2, img.shape[1])
        bb[3] = np.minimum(det[3] + margin / 2, img.shape[0])
        ret = img[bb[1]:bb[3], bb[0]:bb[2], :]
        if len(image_size) > 0:
            ret = cv2.resize(ret, (image_size[1], image_size[0]))
        return ret
    else:  # do align using landmark
        assert len(image_size) == 2

        # src = src[0:3,:]
        # dst = dst[0:3,:]

        # print(src.shape, dst.shape)
        # print(src)
        # print(dst)
        # print(M)
        warped = cv2.warpAffine(img, M, (image_size[1], image_size[0]), borderValue=0.0)

        # tform3 = trans.ProjectiveTransform()
        # tform3.estimate(src, dst)
        # warped = trans.warp(img, tform3, output_shape=_shape)
        return warped
