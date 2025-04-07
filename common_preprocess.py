# 前处理的相关代码
from typing import List, Sequence, Tuple, Union,Optional
import numpy as np
import cv2
import numbers
import torch
import onnxruntime
from collections import namedtuple
import random
import onnxruntime as ort
import os
from PIL import Image

# 常用的前处理方法
'''
PIL转opencv
'''
def pil_to_opencv():
    pil_image = Image.open("image.jpg")  # 或者通过其他方式获取PIL图像
    # 2. 将PIL图像转换为NumPy数组
    # PIL图像通常是RGB格式，而OpenCV使用BGR格式
    opencv_image = np.array(pil_image)

    # 3. 将RGB转换为BGR（如果需要）
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)
    return opencv_image



def opencv_to_pil():
    '''
    opencv转pil
    '''
    opencv_image = cv2.imread("image.jpg")  # OpenCV默认以BGR格式读取图像
    # 2. 将BGR格式转换为RGB格式
    rgb_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    # 3. 将numpy数组转换为PIL图像
    pil_image = Image.fromarray(rgb_image)
    return pil_image

def transpost_opencv(image):
    '''
    opencv换轴
    '''
    image = np.transpose(image, [2, 0, 1])
    return image


def BGR2RGB(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return rgb

def get_image_size(image):
    h, w = image.shape[:2]
    return h,w

def imnormalize(self, img, mean, std, to_rgb=True):
    """Inplace normalize an image with mean and std.

    Args:
        img (ndarray): Image to be normalized.
        mean (ndarray): The mean to be used for normalize.
        std (ndarray): The std to be used for normalize.
        to_rgb (bool): Whether to convert to rgb.

    Returns:
        ndarray: The normalized image.
    """
    mean = np.float64(mean.reshape(1, -1))
    stdinv = 1 / np.float64(std.reshape(1, -1))
    if to_rgb:
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)  # inplace
    cv2.subtract(img, mean, img)  # inplace
    cv2.multiply(img, stdinv, img)  # inplace
    return img

def impad(self, img,
        *,
        shape=None,
        padding=None,
        pad_val=0,
        padding_mode='constant'):
    assert (shape is not None) ^ (padding is not None)
    if shape is not None:
        padding = (0, 0, shape[1] - img.shape[1], shape[0] - img.shape[0])

    # check pad_val
    if isinstance(pad_val, tuple):
        assert len(pad_val) == img.shape[-1]
    elif not isinstance(pad_val, numbers.Number):
        raise TypeError('pad_val must be a int or a tuple. '
                        f'But received {type(pad_val)}')

    # check padding
    if isinstance(padding, tuple) and len(padding) in [2, 4]:
        if len(padding) == 2:
            padding = (padding[0], padding[1], padding[0], padding[1])
    elif isinstance(padding, numbers.Number):
        padding = (padding, padding, padding, padding)
    else:
        raise ValueError('Padding must be a int or a 2, or 4 element tuple.'
                            f'But received {padding}')

    # check padding mode
    assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']

    border_type = {
        'constant': cv2.BORDER_CONSTANT,
        'edge': cv2.BORDER_REPLICATE,
        'reflect': cv2.BORDER_REFLECT_101,
        'symmetric': cv2.BORDER_REFLECT
    }
    # logging.error("padding...")
    # logging.error(padding)
    self.pad_x = padding[0]
    self.pad_y = padding[1]
    img = cv2.copyMakeBorder(
        img,
        padding[1],
        padding[3],
        padding[0],
        padding[2],
        border_type[padding_mode],
        value=pad_val)

    return img,padding

def image_imnormalize(image,means=[123.675,116.28,103.53],stds=[58.395,57.12,57.375]):
    '''
    图像归一化,减均值，除以方差
    '''
    image = np.asarray(image).astype(np.float32)
    mean = np.array(means, dtype=np.float32)
    std = np.array(stds, dtype=np.float32)
    image = imnormalize(image, mean, std)
    image = image[np.newaxis, :, :]
    image = np.ascontiguousarray(image)
    return image

def resize_image(image,resize_width=640,resize_height=640):
    h, w, _ = image.shape
    img_scale = (resize_width, resize_height)
    max_long_edge = max(img_scale)
    max_short_edge = min(img_scale)
    scale_factor = min(max_long_edge / max(h, w), max_short_edge / min(h, w))
    scale_w = int(w * float(scale_factor) + 0.5)
    scale_h = int(h * float(scale_factor) + 0.5)
    image = cv2.resize(image, (scale_w, scale_h))
    image = np.asarray(image).astype(np.float32)
    return image,scale_factor

def letter_box(image,resize_width=640,resize_height=640):
    image = impad(image, shape=(resize_width, resize_height), pad_val=0)
    return image


class onnx_infer():
    def __init__(self):
        self.init_model(device_type='cuda', device_id=0)
        self.scale_factor = 0.5         # 从resize中得到
        self.padding = [0,0,100,100]    # 从impad中得到
        self.image_shape = [640,640]    # 真实的图像大小尺寸


    def init_model(self, device_type='cuda', device_id=0):
        session_options = ort.SessionOptions()
        providers = ['CPUExecutionProvider']
        options = [{}]
        is_cuda_available = ort.get_device() == 'GPU'
        if device_type == 'cuda' and is_cuda_available:
            print('cuda is available')
            providers.insert(0, 'CUDAExecutionProvider')
            options.insert(0, {'device_id': device_id})
            self.sess = ort.InferenceSession(os.path.join(self.file_dir, '../yoloworld_pcba_v14_640.onnx') , session_options, providers=providers, provider_options=options)
            self.output_names = [_.name for _ in self.sess.get_outputs()]
            self.input_name = self.sess.get_inputs()[0].name
            print(self.input_name)
            print(self.output_names)
            self.is_cuda_available = is_cuda_available
        else:
            print('no cuda')

        random_image = np.random.randint(0, 256, (640, 640, 3), dtype=np.uint8)
        self.infer(random_image)


    def onnx_infer(self,image):
        # 输入是四维的numpy
        output_data = self.sess.run(self.output_names, {self.input_name: image})
        result = self.postprocess_nms_numpy(output_data, self.scale_factor, self.padding, self.image_shape, self.iou_threshold, self.threshold, self.pre_topk, self.keep_topk)
        return output_data


    def postprocess_nms_numpy(self, output_data, scale_factor, padding, image_shape, iou_threshold, score_threshold, pre_topk, keep_topk):
        boxes = output_data[0]  # Shape: [batch_size, num_boxes, 4]
        scores = np.max(output_data[1], axis=2)  # Shape: [batch_size, num_boxes]
        labels = np.argmax(output_data[1], axis=2)  # Shape: [batch_size, num_boxes]
        det = np.concatenate([boxes, scores[..., None], labels[..., None]], axis=-1)
        det = det[0]  # Process first image in batch

        # Filter by score threshold
        val_idxs = det[:, 4] > score_threshold
        det = det[val_idxs]

        # Sort by score in descending order
        score_sort = np.argsort(det[:, 4])[::-1]
        det = det[score_sort]

        if len(det) > pre_topk:
            det = det[:pre_topk]

        # Apply NMS per class
        result_boxes = []
        result_scores = []
        result_labels = []

        for label in np.unique(det[:, 5]):
            class_mask = det[:, 5] == label
            class_boxes = det[class_mask][:, :4]
            class_scores = det[class_mask][:, 4]
            
            # Apply NMS for the current class
            keep_idxs = self.nms_numpy(class_boxes, class_scores, iou_threshold)

            # Limit number of boxes to keep_topk
            keep_idxs = keep_idxs[:keep_topk] if len(keep_idxs) > keep_topk else keep_idxs

            if len(keep_idxs) > 0:
                result_boxes.append(class_boxes[keep_idxs])
                result_scores.append(class_scores[keep_idxs])
                result_labels.extend([label] * len(keep_idxs))

        # Only concatenate if result_boxes is not empty
        if len(result_boxes) > 0:
            result_boxes = np.concatenate(result_boxes, axis=0)
            result_scores = np.concatenate(result_scores, axis=0)

            image_h, image_w = image_shape
            final_boxes = []

            for i in range(result_boxes.shape[0]):
                final_boxes.append([
                    max((result_boxes[i][0] - padding[0]) / scale_factor, 0),
                    max((result_boxes[i][1] - padding[1]) / scale_factor, 0),
                    min((result_boxes[i][2] - padding[0]) / scale_factor, image_w - 1),
                    min((result_boxes[i][3] - padding[1]) / scale_factor, image_h - 1)
                ])

            result = {'boxes': final_boxes, 'scores': result_scores.tolist(), 'labels': result_labels}
        else:
            result = {'boxes': [], 'scores': [], 'labels': []}

        return result















