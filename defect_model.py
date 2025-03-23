import numpy as np
import cv2
import numbers
import logging
import math
import time
import torch
import os
import torchvision
from typing import Any, List, Sequence, Tuple, Union, Optional

import sys
if not hasattr(sys, 'argv'):
    sys.argv = ['']
try:
    import onnxruntime as ort
except:
    print('No module named onnxruntime')

try:
    sys.path.append('/usr/local/Ascend/nnrt/latest/python/site-packages/acl')
    sys.path.append('/home/guangcheng-li/userdata/npu_code/samples/python/common/acllite')
    from acllite_resource import AclLiteResource
    from acllite_model import AclLiteModel
    import acl
except:
    print('No module named acllite')

cv2_interp_codes = {
    'nearest': cv2.INTER_NEAREST,
    'bilinear': cv2.INTER_LINEAR,
    'bicubic': cv2.INTER_CUBIC,
    'area': cv2.INTER_AREA,
    'lanczos': cv2.INTER_LANCZOS4 
}


class YOLOv5KeepRatioResize():
    """Resize images & bbox(if existed).

    This transform resizes the input image according to ``scale``.
    Bboxes (if existed) are then resized with the same scale factor.

    Required Keys:

    - img (np.uint8)
    - gt_bboxes (BaseBoxes[torch.float32]) (optional)

    Modified Keys:

    - img (np.uint8)
    - img_shape (tuple)
    - gt_bboxes (optional)
    - scale (float)

    Added Keys:

    - scale_factor (np.float32)

    Args:
        # 这里的scale与H,W顺序无关,只有大小关系！
        scale (Union[int, Tuple[int, int]]): Images scales for resizing.
    """
    def __init__(self,
                    scale: Union[int, Tuple[int, int]],
                    keep_ratio: bool = True,
                    **kwargs):
        assert keep_ratio is True
        self.keep_ratio = keep_ratio
        self.scale = scale

    def _get_rescale_ratio(self, old_size: Tuple[int, int],
                            scale: Union[float, Tuple[int]]) -> float:
        """Calculate the ratio for rescaling.

        Args:
            old_size (tuple[int]): The old size (w, h) of image.
            scale (float | tuple[int]): The scaling factor or maximum size.
                If it is a float number, then the image will be rescaled by
                this factor, else if it is a tuple of 2 integers, then
                the image will be rescaled as large as possible within
                the scale.

        Returns:
            float: The resize ratio.
        """
        w, h = old_size
        if isinstance(scale, (float, int)):
            if scale <= 0:
                raise ValueError(f'Invalid scale {scale}, must be positive.')
            scale_factor = scale
        elif isinstance(scale, tuple):
            max_long_edge = max(scale)
            max_short_edge = min(scale)
            scale_factor = min(max_long_edge / max(h, w),
                                max_short_edge / min(h, w))
        else:
            raise TypeError('Scale must be a number or tuple of int, '
                            f'but got {type(scale)}')

        return scale_factor

    def _resize_img(self, result):
        """Resize images with ``scale``."""
        assert self.keep_ratio is True

        image = result.get('img', None)
        if image is None:
            return

        original_h, original_w = image.shape[:2]
        ratio = self._get_rescale_ratio((original_h, original_w), self.scale)

        if ratio != 1:
            
            image = cv2.resize(
                image,
                (int(original_w * ratio), int(original_h * ratio)),
                interpolation=cv2_interp_codes['area' if ratio < 1 else 'bilinear'])

        resized_h, resized_w = image.shape[:2]
        scale_ratio_h = resized_h / original_h
        scale_ratio_w = resized_w / original_w
        scale_factor = (scale_ratio_w, scale_ratio_h)

        result['img'] = image
        result['ori_img_shape'] = original_h, original_w
        result['scale'] = self.scale
        result['img_shape'] = image.shape[:2]
        result['scale_factor'] = scale_factor
    
    def transform(self, result, *args: Any, **kwds: Any) -> Any:
        self._resize_img(result)
        return result
    

class MMYoloLetterResize():
    """Resize and pad image while meeting stride-multiple constraints.

    Required Keys:

    - img (np.uint8)
    - batch_shape (np.int64) (optional)

    Modified Keys:

    - img (np.uint8)
    - img_shape (tuple)
    - gt_bboxes (optional)

    Added Keys:
    - pad_param (np.float32)

    Args:
        scale (Union[int, Tuple[int, int]]): Images scales for resizing.
        pad_val (dict): Padding value. Defaults to dict(img=0, seg=255).
        use_mini_pad (bool): Whether using minimum rectangle padding.
            Defaults to True
        stretch_only (bool): Whether stretch to the specified size directly.
            Defaults to False
        allow_scale_up (bool): Allow scale up when ratio > 1. Defaults to True
        half_pad_param (bool): If set to True, left and right pad_param will
            be given by dividing padding_h by 2. If set to False, pad_param is
            in int format. We recommend setting this to False for object
            detection tasks, and True for instance segmentation tasks.
            Default to False.
    """

    def __init__(self,
                 scale: Union[int, Tuple[int, int]],
                 pad_val: dict = dict(img=0, mask=0, seg=255),
                 use_mini_pad: bool = False,
                 stretch_only: bool = False,
                 allow_scale_up: bool = True,
                 half_pad_param: bool = False,
                 **kwargs):

        # H, W required
        self.scale = scale
        self.keep_ratio = True
        
        self.pad_val = pad_val
        if isinstance(pad_val, (int, float)):
            pad_val = dict(img=pad_val, seg=255)
        assert isinstance(
            pad_val, dict), f'pad_val must be dict, but got {type(pad_val)}'

        self.use_mini_pad = use_mini_pad
        self.stretch_only = stretch_only
        self.allow_scale_up = allow_scale_up
        self.half_pad_param = half_pad_param

    def _resize_img(self, results: dict):
        """Resize images with ``results['scale']``."""
        image = results.get('img', None)
        if image is None:
            return

        # H, W
        scale = self.scale

        image_shape = image.shape[:2]  # height, width
        
        # Scale ratio (new / old)
        ratio = min(scale[0] / image_shape[0], scale[1] / image_shape[1])

        # only scale down, do not scale up (for better test mAP)
        if not self.allow_scale_up:
            ratio = min(ratio, 1.0)

        ratio = [ratio, ratio]  # float -> (float, float) for (height, width)

        # compute the best size of the image
        no_pad_shape = (int(round(image_shape[0] * ratio[0])),
                        int(round(image_shape[1] * ratio[1])))

        # padding height & width
        padding_h, padding_w = [
            scale[0] - no_pad_shape[0], scale[1] - no_pad_shape[1]
        ]
        if self.use_mini_pad:
            # minimum rectangle padding
            padding_w, padding_h = np.mod(padding_w, 32), np.mod(padding_h, 32)

        elif self.stretch_only:
            # stretch to the specified size directly
            padding_h, padding_w = 0.0, 0.0
            no_pad_shape = (scale[0], scale[1])
            ratio = [scale[0] / image_shape[0],
                     scale[1] / image_shape[1]]  # height, width ratios

        if image_shape != no_pad_shape:
            # compare with no resize and padding size
            image = cv2.resize(
                image, (no_pad_shape[1], no_pad_shape[0]),
                interpolation=self.interpolation)
        
        scale_factor = (no_pad_shape[1] / image_shape[1],
                        no_pad_shape[0] / image_shape[0])

        if 'scale_factor' in results:
            results['scale_factor_origin'] = results['scale_factor']
        results['scale_factor'] = scale_factor

        # padding
        top_padding, left_padding = int(round(padding_h // 2 - 0.1)), int(
            round(padding_w // 2 - 0.1))
        bottom_padding = padding_h - top_padding
        right_padding = padding_w - left_padding

        padding_list = [
            top_padding, bottom_padding, left_padding, right_padding
        ]
        if top_padding != 0 or bottom_padding != 0 or \
                left_padding != 0 or right_padding != 0:

            pad_val = self.pad_val.get('img', 0)
            if isinstance(pad_val, int) and image.ndim == 3:
                pad_val = tuple(pad_val for _ in range(image.shape[2]))
                
            image = mmcv_impad(
                img=image,
                padding=(padding_list[2], padding_list[0], padding_list[3],
                         padding_list[1]),
                pad_val=pad_val,
                padding_mode='constant')

        results['img'] = image
        results['img_shape'] = image.shape
        if 'pad_param' in results:
            results['pad_param_origin'] = results['pad_param'] * \
                                          np.repeat(ratio, 2)

        if self.half_pad_param:
            results['pad_param'] = np.array(
                [padding_h / 2, padding_h / 2, padding_w / 2, padding_w / 2],
                dtype=np.float32)
        else:
            # We found in object detection, using padding list with
            # int type can get higher mAP.
            results['pad_param'] = np.array(padding_list, dtype=np.float32)

    def transform(self, results: dict) -> dict:
        self._resize_img(results)
        if 'scale_factor_origin' in results:
            scale_factor_origin = results.pop('scale_factor_origin')
            results['scale_factor'] = (results['scale_factor'][0] *
                                       scale_factor_origin[0],
                                       results['scale_factor'][1] *
                                       scale_factor_origin[1])
        if 'pad_param_origin' in results:
            pad_param_origin = results.pop('pad_param_origin')
            results['pad_param'] += pad_param_origin
        return results
    
def mmcv_impad(img: np.ndarray,
          *,
          shape: Optional[Tuple[int, int]] = None,
          padding: Union[int, tuple, None] = None,
          pad_val: Union[float, List] = 0,
          padding_mode: str = 'constant') -> np.ndarray:
    """Pad the given image to a certain shape or pad on all sides with
    specified padding mode and padding value.

    Args:
        img (ndarray): Image to be padded.
        shape (tuple[int]): Expected padding shape (h, w). Default: None.
        padding (int or tuple[int]): Padding on each border. If a single int is
            provided this is used to pad all borders. If tuple of length 2 is
            provided this is the padding on left/right and top/bottom
            respectively. If a tuple of length 4 is provided this is the
            padding for the left, top, right and bottom borders respectively.
            Default: None. Note that `shape` and `padding` can not be both
            set.
        pad_val (Number | Sequence[Number]): Values to be filled in padding
            areas when padding_mode is 'constant'. Default: 0.
        padding_mode (str): Type of padding. Should be: constant, edge,
            reflect or symmetric. Default: constant.

            - constant: pads with a constant value, this value is specified
              with pad_val.
            - edge: pads with the last value at the edge of the image.
            - reflect: pads with reflection of image without repeating the last
              value on the edge. For example, padding [1, 2, 3, 4] with 2
              elements on both sides in reflect mode will result in
              [3, 2, 1, 2, 3, 4, 3, 2].
            - symmetric: pads with reflection of image repeating the last value
              on the edge. For example, padding [1, 2, 3, 4] with 2 elements on
              both sides in symmetric mode will result in
              [2, 1, 1, 2, 3, 4, 4, 3]

    Returns:
        ndarray: The padded image.
    """

    assert (shape is not None) ^ (padding is not None)
    if shape is not None:
        width = max(shape[1] - img.shape[1], 0)
        height = max(shape[0] - img.shape[0], 0)
        padding = (0, 0, width, height)

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
    img = cv2.copyMakeBorder(
        img,
        padding[1],
        padding[3],
        padding[0],
        padding[2],
        border_type[padding_mode],
        value=pad_val)
    return img


def bbox_iou(bbox1, bbox2):
    "bbox: (x0,y0,x1,y1)"
    left_column_max  = max(bbox1[0], bbox2[0])
    right_column_min = min(bbox1[2], bbox2[2])
    up_row_max       = max(bbox1[1], bbox2[1])
    down_row_min     = min(bbox1[3], bbox2[3])
    #两矩形无相交区域的情况
    if left_column_max >= right_column_min or down_row_min <= up_row_max:
        return 0
    # 两矩形有相交区域的情况
    else:
        S1 = (bbox1[2]-bbox1[0])*(bbox1[3]-bbox1[1])
        S2 = (bbox2[2]-bbox2[0])*(bbox2[3]-bbox2[1])
        S_cross = (down_row_min-up_row_max)*(right_column_min-left_column_max)
        return S_cross/(S1+S2-S_cross)


def get_cosine_similarity(vec1, vec2):
    cos_sim = vec1.dot(vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return cos_sim

# ENMS
def get_entropy_by_enms(classes, scores, instance_features, threshold=0.5, filter_inds=[]):
    i_entropy = 0
    entropys = -scores*np.log(scores+0.0000001)-(1-scores)*np.log(1-scores+0.0000001)
    # classes, scores, instance_features, entropys = classes[0], scores[0], instance_features[0], entropys[0]

    # 过滤特定下标
    remove_index = []
    for ind in range(len(entropys)):
        if ind in filter_inds:
            remove_index.append(ind)
    classes = np.delete(classes, remove_index)
    scores = np.delete(scores, remove_index)
    instance_features = np.delete(instance_features, remove_index, axis=0)
    entropys = np.delete(entropys, remove_index)

    while len(entropys):
        pick = np.argmax(entropys)
        c_pick = classes[pick]
        f_pick = instance_features[pick]
        e_pick = entropys[pick]
        
        classes = np.delete(classes, pick)
        scores = np.delete(scores, pick)
        instance_features = np.delete(instance_features, pick, axis=0)
        entropys = np.delete(entropys, pick)
        i_entropy += e_pick

        remove_index = []
        for j in range(len(entropys)):
            if classes[j] == c_pick and get_cosine_similarity(instance_features[j], f_pick) > threshold:
                remove_index.append(j)

        classes = np.delete(classes, remove_index)
        scores = np.delete(scores, remove_index)
        instance_features = np.delete(instance_features, remove_index, axis=0)
        entropys = np.delete(entropys, remove_index)
    return i_entropy


# LNMS
def get_lloss_by_lnms(classes, llosses, instance_features, threshold=0.5, filter_inds=[]):
    i_lloss = 0
    llosses = np.exp(llosses)
    # classes, llosses, instance_features = classes[0], llosses[0], instance_features[0]

    # 过滤特定类别
    remove_index = []
    for ind in range(len(llosses)):
        if ind in filter_inds:
            remove_index.append(ind)
    classes = np.delete(classes, remove_index)
    llosses = np.delete(llosses, remove_index)
    instance_features = np.delete(instance_features, remove_index, axis=0)

    while len(llosses):
        pick = np.argmax(llosses)
        c_pick = classes[pick]
        l_pick = llosses[pick]
        f_pick = instance_features[pick]

        classes = np.delete(classes, pick)
        llosses = np.delete(llosses, pick)
        instance_features = np.delete(instance_features, pick, axis=0)
        i_lloss += l_pick

        remove_index = []
        for j in range(len(llosses)):
            if classes[j] == c_pick and get_cosine_similarity(instance_features[j], f_pick) > threshold:
                remove_index.append(j)

        classes = np.delete(classes, remove_index)
        llosses = np.delete(llosses, remove_index)
        instance_features = np.delete(instance_features, remove_index, axis=0)
        
    return i_lloss


def foramt_outputs_with_nms(outputs, iou_threshold=0.6, max_per_img=1000, max_after_img=100):
    boxes = outputs['dets']
    scores = outputs['labels']
    lloss = outputs['learning_loss']
    features = outputs['entropy']
    
    batch_size = scores.shape[0]
    num_box = scores.shape[1]
    num_class = scores.shape[2]
    features_dim = features.shape[2]

    max_per_img = min(max_per_img, boxes.shape[1])
    topk_inds = scores.max(-1).argsort()[:,::-1][:,:max_per_img]

    batch_inds = np.arange(batch_size).reshape(-1, 1).repeat(max_per_img, axis=1)
    transformed_inds = num_box * batch_inds + topk_inds
    boxes = boxes.reshape(-1, 4)[transformed_inds, :].reshape(
            batch_size, -1, 4)
    scores = scores.reshape(-1, num_class)[transformed_inds, :].reshape(
            batch_size, -1, num_class)
    lloss = lloss.reshape(-1, 1)[transformed_inds, :].reshape(
        batch_size, -1)
    features = features.reshape(-1, features_dim)[transformed_inds, :].reshape(
        batch_size, -1, features_dim)

    num_box = scores.shape[1]
    labels = scores.argmax(axis=-1)
    for batch_index in range(batch_size):
        deleted_flags = np.ones((num_box, num_class))
        for i in range(num_box):
            if scores[batch_index][i].max() < 0.1:
                break
            if deleted_flags[i].any() == False:
                continue
            for j in range(i+1, num_box):
                if labels[batch_index][i]==labels[batch_index][j] and bbox_iou(boxes[batch_index][i], boxes[batch_index][j]) > iou_threshold:
                    deleted_flags[j][::] = 0
        
        scores[batch_index] = scores[batch_index] * deleted_flags

    num_box = scores.shape[1]
    max_after_img = min(max_after_img, boxes.shape[1])
    topk_inds = scores.max(-1).argsort()[:,::-1][:,:max_after_img]

    batch_inds = np.arange(batch_size).reshape(-1, 1).repeat(max_after_img, axis=1)
    transformed_inds = num_box * batch_inds + topk_inds
    boxes = boxes.reshape(-1, 4)[transformed_inds, :].reshape(
            batch_size, -1, 4)
    scores = scores.reshape(-1, num_class)[transformed_inds, :].reshape(
            batch_size, -1, num_class)
    lloss = lloss.reshape(-1, 1)[transformed_inds, :].reshape(
        batch_size, -1)
    features = features.reshape(-1, features_dim)[transformed_inds, :].reshape(
        batch_size, -1, features_dim)

    labels = scores.argmax(axis=-1)
    scores = scores.max(axis=-1, keepdims=True)
    dets = np.concatenate([boxes, scores], axis=2)

    outputs['dets'] = dets
    outputs['labels'] = labels
    outputs['learning_loss'] = lloss
    outputs['entropy'] = features
    return outputs


def foramt_outputs(outputs, max_per_img=100):
    boxes = outputs['dets']
    scores = outputs['labels']
    lloss = outputs['learning_loss']
    features = outputs['entropy']
    

    max_per_img = min(max_per_img, boxes.shape[1])
    boxes = boxes[:,:max_per_img,:]
    scores = scores[:,:max_per_img,:]
    lloss = lloss[:,:max_per_img,:]
    features = features[:,:max_per_img,:]
    
    batch_size = scores.shape[0]
    num_box = scores.shape[1]
    num_class = scores.shape[2]

    labels = scores.argmax(axis=-1)
    batch_inds = np.arange(batch_size).reshape(-1, 1).repeat(num_box, axis=1)
    box_inds = np.arange(num_box).reshape(1, -1).repeat(batch_size, axis=0)
    transformed_inds = num_box * num_class * batch_inds + num_class * box_inds + labels
    scores = scores.reshape(-1, 1)[transformed_inds, :].reshape(batch_size, -1, 1)

    dets = np.concatenate([boxes, scores], axis=2)
    outputs['dets'] = dets
    outputs['labels'] = labels
    outputs['learning_loss'] = lloss
    outputs['entropy'] = features

    return outputs


class DefectModel:
    # 根据需要添加其它函数
    def imnormalize(self, img, mean=[0.,0.,0.], std=[255.,255.,255.], to_rgb=True):
        """Inplace normalize an image with mean and std.
    
        Args:
            img (ndarray): Image to be normalized.
            mean (ndarray): The mean to be used for normalize.
            std (ndarray): The std to be used for normalize.
            to_rgb (bool): Whether to convert to rgb.
    
        Returns:
            ndarray: The normalized image.
        """
        # cv2 inplace normalization does not accept uint8
        #assert img.dtype != np.uint8
        mean = np.array(mean)
        std = np.array(std)
        mean = np.float64(mean.reshape(1, -1))
        stdinv = 1 / np.float64(std.reshape(1, -1))
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

        return img


    def __init__(self, resize=(416, 416), score_threshold=0.1, iou_threshold=0.3, batch_size=1, pre_topk=1000,keep_topk=100,device='cuda:0'):
        self.file_dir = os.path.dirname(os.path.abspath(__file__))
        self.means = [0,0,0]
        self.stds = [255.,255.,255.]
        self.resize_width, self.resize_height = resize
        self.class_name =['侧立', '偏移', '反向', '反白', '少件', '少锡', '异物', '引脚错位', '损件', '短路', '立碑', '翘脚', '错件', '锡珠', '虚焊']
        self.threshold = score_threshold
        self.iou_threshold = iou_threshold
        self.infer_max_batch_size = batch_size
        self.pre_topk = pre_topk
        self.keep_topk = keep_topk

        self.KeepRatioResize = YOLOv5KeepRatioResize(resize)
        self.LetterResize = MMYoloLetterResize(resize,allow_scale_up=False,pad_val=dict(img=114))
        
        self.device_type, device_id = device.split(':')
        self.device_id = int(device_id)
        self.init_model(device_type=self.device_type, device_id=self.device_id)


    # 初始化模型
    def init_model(self, device_type='cuda', device_id=0):
        if device_type == 'cuda':
            session_options = ort.SessionOptions()
            providers = ['CPUExecutionProvider']
            options = [{}]
            is_cuda_available = ort.get_device() == 'GPU'
            if is_cuda_available:
                providers.insert(0, 'CUDAExecutionProvider')
                options.insert(0, {'device_id': self.device_id})
            sess = ort.InferenceSession(os.path.join(self.file_dir, '../model_0318_fp16.onnx'), session_options, providers=providers, provider_options=options)
            self.sess = sess
            self.io_binding = sess.io_binding()
            self.output_names = [_.name for _ in sess.get_outputs()]
            self.is_cuda_available = is_cuda_available
        elif device_type =='npu':
            self.model = AclLiteModel(os.path.join(self.file_dir, '../OCR.om'))

    def preprocess(self, inputs, meta_list):
        images = []
        preprocess_time = []
        for image, meta in zip(inputs, meta_list):
            t1 = time.time()
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            image_info = {'img': rgb}
            image_info = self.KeepRatioResize.transform(image_info)
            image_info = self.LetterResize.transform(image_info)
            pad_param = image_info.get('pad_param',
                                    np.array([0, 0, 0, 0], dtype=np.float32))            
            meta['pad_param'] = np.array([pad_param[2], pad_param[0], pad_param[2], pad_param[0]], dtype=np.float32)
            scale_factor = image_info.get('scale_factor', [1., 1])
            meta['scale_factor'] = np.array([scale_factor * 2], dtype=np.float32)
            meta['w'] = w
            meta['h'] = h
            
            # todo: check dimension
            image = np.asarray(image_info['img']).astype(np.float32)
            image = self.imnormalize(image, self.means, self.stds)
            image = np.ascontiguousarray(np.transpose(image, [2, 0, 1]))
            images.append(image)
            t2 = time.time()
            preprocess_time.append(t2-t1)
            print(f'preprocess_time:{preprocess_time}')
        return {'input': images},preprocess_time
    
    
    def postprocess(self, images, inputs, meta_list):
        results = []
        
        # inputs = foramt_outputs_with_nms(inputs)
        batched_bboxes = inputs['bboxes']
        batched_scores = np.max(inputs['scores'], 2)
        batched_labels = np.argmax(inputs['scores'], 2)
        batched_det = np.c_[batched_bboxes, batched_scores[..., None], batched_labels[..., None]]

        # 取每张图片的结果
        for meta_index, (image, det, meta) in enumerate(zip(images, batched_det, meta_list)):
            bboxes = []
            attributes = {}
            image_h, image_w, _ = image.shape
            entropy = 0
            # nms
            # det: num*(4+1+1) bboxes, scores, labels
            ValIdxs = det[:, 4] > self.threshold
            det = det[ValIdxs]
            ScoreSort = np.argsort(det[:, 4])[::-1]
            det = det[ScoreSort]
            
            if len(det) > self.pre_topk:
                det = det[:self.pre_topk]
            
            # cls ignore
            # keep_idxs = torchvision.ops.batched_nms(torch.from_numpy(det[..., :4]), torch.from_numpy(det[..., 4]), torch.from_numpy(det[..., 5]), self.iou_threshold)
            keep_idxs = torchvision.ops.nms(torch.from_numpy(det[..., :4]), torch.from_numpy(det[..., 5]), self.iou_threshold)

            if len(keep_idxs) > self.keep_topk:
                keep_idxs = keep_idxs[:self.keep_topk]
            
            num_keep_idxs = len(keep_idxs)
            if num_keep_idxs:
                det = det[keep_idxs]
                if num_keep_idxs == 1:
                    det = det[None]
            else:
                det = np.zeros((0,5))
                label = np.zeros((0))
            
            # traceback to original image
            if len(det):
                det[..., :4] -= meta['pad_param']
                det[..., :4] /= meta['scale_factor']
                det[..., 0:4:2] = np.clip(det[..., 0:4:2], 0, meta['w'])
                det[..., 1:4:2] = np.clip(det[..., 1:4:2], 0, meta['h'])
                dets = det[..., :5]
                labels = det[..., 5]
            else:                
                dets = det
                labels = label

            entropy = 0
            score = 0
            label_name = 'OK'
            for ind, (det, index) in enumerate(zip(dets, labels)):
                # 计算熵
                det = det.tolist()
                index = int(index)
                entropy += (-det[4]*math.log(det[4]+0.0000001)-(1-det[4])*math.log(1-det[4]+0.0000001))
                if det[2]-det[0]>2 and det[3]-det[1]>2:
                    if ind == 0:
                        score = det[4]
                        label_name = self.class_name[index]

                    bbox_cx = int(0.5*(det[0] + det[2]))
                    bbox_cy = int(0.5*(det[1] + det[3]))
                    centerX = int(image_w/2)
                    centerY = int(image_h/2)
                    # if self.class_name[index]=='少件' and (bbox_cx < centerX-0.3*image_w or bbox_cx > centerX+0.3*image_w or bbox_cy < centerY-0.3*image_h or bbox_cy > centerY+0.3*image_h):
                    #     continue
                    
                    # print(self.class_name[index],det[4])

                    bboxes.append([max(det[0], 0),
                                max(det[1], 0),
                                min(det[2], image_w-1),
                                min(det[3], image_h-1),
                                det[4],
                                self.class_name[index],
                                index])
                
            attributes['score'] = float(score)
            attributes['label_name'] = label_name

            attributes['entropy'] = float(entropy)
            attributes['learning_loss'] = float(0)
            attributes['entropy_nms'] = float(0)
            attributes['learning_loss_nms'] = float(0)

            results.append({
                'bboxes': bboxes,
                'attributes': attributes
            })
        
        return results

        
    def infer(self, inputs, meta_list):
        # results = {'dets':[], 'labels':[], 'feature':[], 'entropy':[], 'learning_loss':[]}
        results = {'bboxes':[], 'scores':[]}
        if self.device_type == 'cuda':
            images = [torch.from_numpy(image).unsqueeze(0).half() for image in inputs['input']]
            index = 0
            num = len(images)
            infer_time = []
            while index < num:
                t1 = time.time()
                index += self.infer_max_batch_size
                if index <= num:
                    one_img = torch.cat(images[index-self.infer_max_batch_size:index], dim=0)
                    one_meta = [{'scale_factor': np.ones(4, dtype=np.float)} for meta in meta_list[index-self.infer_max_batch_size:index]]
                else:
                    one_img = torch.cat(images[index-self.infer_max_batch_size:num], dim=0)
                    one_meta = [{'scale_factor': np.ones(4, dtype=np.float)} for meta in meta_list[index-self.infer_max_batch_size:num]]
                img_list, img_meta_list = [one_img], [one_meta]
                img_list = [_.cuda(self.device_id).contiguous() for _ in img_list]


                input_data = img_list[0]
                # set io binding for inputs/outputs
                device_type = 'cuda' if self.is_cuda_available else 'cpu'
                if not self.is_cuda_available:
                    input_data = input_data.cpu()
                self.io_binding.bind_input(
                    name='images',
                    device_type=device_type,
                    device_id=self.device_id,
                    element_type=np.float16,
                    shape=input_data.shape,
                    buffer_ptr=input_data.data_ptr())

                for name in self.output_names:
                    self.io_binding.bind_output(name)
                # run session to get outputs
                self.sess.run_with_iobinding(self.io_binding)
                outputs = self.io_binding.copy_outputs_to_cpu()

                results['bboxes'].append(outputs[0])
                results['scores'].append(outputs[1])

                t2 = time.time()
                infer_time.append(t2-t1)

        elif self.device_type == 'npu':
            images = [np.ascontiguousarray(np.expand_dims(image, axis=0)) for image in inputs['input']]
            # 每张图片进行推理
            infer_time = []
            t1 = time.time()
            for image in images:
                outputs = self.model.execute(image)
                results['bboxes'].append(outputs[0])
                results['scores'].append(outputs[1])
            t2 = time.time()
            infer_time.append(t2-t1)
            # index = 0
            # num = len(images)
            # while index < num:
                # index += self.infer_max_batch_size
                # if index <= num:
                    # one_img = np.concatenate(images[index-self.infer_max_batch_size:index], axis=0)
                # else:
                    # one_img = np.concatenate(images[index-self.infer_max_batch_size:num], axis=0)
                    # if num - (index-self.infer_max_batch_size) < self.infer_max_batch_size:
                        # full0 =self.infer_max_batch_size - (num - (index-self.infer_max_batch_size))
                        # 创建一张空图像
                        # ImageZero = np.zeros((full0, 3, 416, 416), dtype=np.uint8)
                        # one_img = np.concatenate((one_img,ImageZero),axis=0)
                # outputs = self.model._execute_with_dynamic_batch_size([one_img], self.infer_max_batch_size)
                # results['dets'].append(outputs[0])
                # results['labels'].append(outputs[1])
                # results['feature'].append(outputs[2])
                # results['entropy'].append(outputs[3])
                # results['learning_loss'].append(outputs[4])
        # 未做NMS
        for key, value in results.items():
            results[key] = np.concatenate(results[key], axis=0)
        return results,infer_time

    def pipeline(self, inputs):
        meta_list = [{} for _ in inputs]
        # 缩放图像大小，转换图片格式
        images,preprocess_time = self.preprocess(inputs, meta_list)
        # 推理
        output,infer_time = self.infer(images, meta_list)
        # 后处理
        result = self.postprocess(inputs, output, meta_list)
        return result,preprocess_time,infer_time
    
    # 可选提供finalize函数
    def finalize(self):
        pass


if __name__ == '__main__':
    import time
    from PIL import Image, ImageDraw, ImageFont

    # resource = AclLiteResource()
    # resource.init()
    
    batch_size = 1
    model = DefectModel(resize=(640, 640), score_threshold=0.2, iou_threshold=0.25, batch_size=batch_size,pre_topk=1000,keep_topk=100, device='cuda:0')
    #
    if model.device_type == 'cuda':
        output_dir = './cuda_outputs'
    elif model.device_type == 'npu':
        output_dir = '../../../npu_outputs'
    os.makedirs(output_dir, exist_ok=False)

    images_dir = '/home/apulis-test/userdata/code/jingshi-project/images/normal_analy/0310_0320/xuhan/comp'

    image_files = os.listdir(images_dir)

    all_time_list = []
    all_num_list = []

    pred_ok_num = 0
    pred_ng_num = 0

    num = len(image_files)
    index = 0
    while index < num:
        print(f'{index}/{num}')
        index += batch_size
        if index <= num:
            batch_image_files = image_files[index-batch_size:index]
        else:
            batch_image_files = image_files[index-batch_size:num]

        inputs = []
        components = []
        for image_file in batch_image_files:
            image_path = os.path.join(images_dir, image_file)
            print(f"image_path:{image_path}")
            image = cv2.imread(image_path)
            inputs.append(image)
        t1 = time.time()
        if inputs[0] is None:
            continue
        results,_,_ = model.pipeline(inputs)
        print(f'results-->{results}')
        t2 = time.time()
        print('pipeline time: ', t2 - t1)
        print('result!!!!')
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        all_time_list.append(t2 - t1)
        all_num_list.append(len(results))
        print(results[0]['attributes']['label_name'])


        for image_file, image, result in zip(batch_image_files, inputs, results):
            for x0, y0, x1, y1, score, label_name, label_id  in result['bboxes']:
                s = f'{label_name} | {score:.2f}'
                cv2.rectangle(image, (int(x0), int(y0)), (int(x1), int(y1)), (0,0,255), 2)
                image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(image)
                fontStyle = ImageFont.truetype('/home/apulis-test/userdata/code/script/simsun.ttc', 25, encoding="utf-8")
                draw.text( (int(x0), int(y0) + 25), s, (255, 255, 255), font=fontStyle)
                image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

            if not os.path.exists(os.path.join(output_dir,'ok')):
                os.mkdir(os.path.join(output_dir,'ok'))
            if not os.path.exists(os.path.join(output_dir,'ng')):
                os.mkdir(os.path.join(output_dir,'ng'))
            if not os.path.exists(os.path.join(output_dir,'ng',results[0]['attributes']['label_name'])):
                os.mkdir(os.path.join(output_dir,'ng',results[0]['attributes']['label_name']))

            if len(result['bboxes']) == 0:
                cv2.imwrite(os.path.join(output_dir,'ok', image_file), image)
            else:
                cv2.imwrite(os.path.join(output_dir,'ng',results[0]['attributes']['label_name'],image_file), image)
        # exit()

    print('image file num: ', len(image_files))
    print('all time: ', sum(all_time_list))
    if len(image_files) > 2:
        print('avg fps: ', sum(all_num_list[1:-1])/sum(all_time_list[1:-1]))