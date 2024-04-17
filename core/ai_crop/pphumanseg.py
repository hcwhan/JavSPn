# This file is part of OpenCV Zoo project.
# It is subject to the license terms in the LICENSE file found in the same directory.
#
# Copyright (C) 2021, Shenzhen Institute of Artificial Intelligence and Robotics for Society, all rights reserved.
# Third party copyrights are property of their respective owners.

import numpy as np
import cv2 as cv

# from core.config import cfg

class PPHumanSeg:
    def __init__(self, modelPath, backendId=0, targetId=0):
        self._modelPath = modelPath
        self._backendId = backendId
        self._targetId = targetId

        self._model = cv.dnn.readNet(self._modelPath)
        self._model.setPreferableBackend(self._backendId)
        self._model.setPreferableTarget(self._targetId)

        self._inputNames = ''
        self._outputNames = ['save_infer_model/scale_0.tmp_1']
        self._currentInputSize = None
        self._inputSize = [192, 192]
        self._mean = np.array([0.5, 0.5, 0.5])[np.newaxis, np.newaxis, :]
        self._std = np.array([0.5, 0.5, 0.5])[np.newaxis, np.newaxis, :]

    @property
    def name(self):
        return self.__class__.__name__

    def setBackendAndTarget(self, backendId, targetId):
        self._backendId = backendId
        self._targetId = targetId
        self._model.setPreferableBackend(self._backendId)
        self._model.setPreferableTarget(self._targetId)

    def _preprocess(self, image):

        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        self._currentInputSize = image.shape
        image = cv.resize(image, (192, 192))
        
        image = image.astype(np.float32, copy=False) / 255.0
        image -= self._mean
        image /= self._std
        return cv.dnn.blobFromImage(image)

    def infer(self, image):

        # Preprocess
        inputBlob = self._preprocess(image)

        # Forward
        self._model.setInput(inputBlob, self._inputNames)
        outputBlob = self._model.forward()

        # Postprocess
        results = self._postprocess(outputBlob)

        return results

    def _postprocess(self, outputBlob):
        
        outputBlob = outputBlob[0]
        outputBlob = cv.resize(outputBlob.transpose(1,2,0), (self._currentInputSize[1], self._currentInputSize[0]), interpolation=cv.INTER_LINEAR).transpose(2,0,1)[np.newaxis, ...]

        result = np.argmax(outputBlob, axis=1).astype(np.uint8)
        return result

# piccfg = cfg.Picture
# if piccfg == None:
#     raise Exception("Missing `Picture` field from config.ini")
# model_path = piccfg.pphumanseg_model
# if model_path == None:
#     raise Exception("Missing `pphumanseg_model` for ai crop backend pphumanseg")
model_path = "/home/lyh/Downloads/human_segmentation_pphumanseg_2023mar.onnx"

# Valid combinations of backends and targets
backend_target_pairs = [
    [cv.dnn.DNN_BACKEND_OPENCV, cv.dnn.DNN_TARGET_CPU],
    [cv.dnn.DNN_BACKEND_CUDA,   cv.dnn.DNN_TARGET_CUDA],
    [cv.dnn.DNN_BACKEND_CUDA,   cv.dnn.DNN_TARGET_CUDA_FP16],
    [cv.dnn.DNN_BACKEND_TIMVX,  cv.dnn.DNN_TARGET_NPU],
    [cv.dnn.DNN_BACKEND_CANN,   cv.dnn.DNN_TARGET_NPU]
]

backend_target = 0
backend_id = backend_target_pairs[backend_target][0]
target_id = backend_target_pairs[backend_target][1]

# def get_color_map_list(num_classes):
#     """
#     Returns the color map for visualizing the segmentation mask,
#     which can support arbitrary number of classes.
#
#     Args:
#         num_classes (int): Number of classes.
#
#     Returns:
#         (list). The color map.
#     """
#
#     num_classes += 1
#     color_map = num_classes * [0, 0, 0]
#     for i in range(0, num_classes):
#         j = 0
#         lab = i
#         while lab:
#             color_map[i * 3] |= (((lab >> 0) & 1) << (7 - j))
#             color_map[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j))
#             color_map[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j))
#             j += 1
#             lab >>= 3
#     color_map = color_map[3:]
#     return color_map
#
# def visualize(image, result, weight=0.6, fps=None):
#     """
#     Convert predict result to color image, and save added image.
#
#     Args:
#         image (str): The input image.
#         result (np.ndarray): The predict result of image.
#         weight (float): The image weight of visual image, and the result weight is (1 - weight). Default: 0.6
#         fps (str): The FPS to be drawn on the input image.
#
#     Returns:
#         vis_result (np.ndarray): The visualized result.
#     """
#     color_map = get_color_map_list(256)
#     color_map = np.array(color_map).reshape(256, 3).astype(np.uint8)
#
#     # Use OpenCV LUT for color mapping
#     c1 = cv.LUT(result, color_map[:, 0])
#     c2 = cv.LUT(result, color_map[:, 1])
#     c3 = cv.LUT(result, color_map[:, 2])
#     pseudo_img = np.dstack((c1, c2, c3))
#
#     vis_result = cv.addWeighted(image, weight, pseudo_img, 1 - weight, 0)
#
#     if fps is not None:
#         cv.putText(vis_result, 'FPS: {:.2f}'.format(fps), (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
#
#     return vis_result


model = PPHumanSeg(modelPath=model_path, backendId=backend_id, targetId=target_id)

def ai_crop_poster(fanart, poster='', hw_ratio=1.42):
    image = cv.imread(fanart)
    fanart_h, fanart_w, _ = image.shape
    poster_h = fanart_h
    poster_w = fanart_h / hw_ratio 
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    _image = cv.resize(image, dsize=(192, 192))

    # Inference
    result = model.infer(_image)
    result = cv.resize(result[0, :, :], dsize=(fanart_w, fanart_h), interpolation=cv.INTER_NEAREST)

    indicies_y, indicies_x = np.nonzero(result)
    center_x = np.average(indicies_x)
    poster_left = max(center_x - poster_w / 2, 0)
    poster_left = min(poster_left, fanart_w - poster_w)
    poster_left = int(poster_left)
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    cropped = image[0:poster_h, poster_left:int(poster_left+poster_w)]
    cv.imwrite(poster, cropped) 

if __name__ == "__main__":
    import sys
    ai_crop_poster(sys.argv[1])
