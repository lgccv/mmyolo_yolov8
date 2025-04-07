# 前处理的相关代码
from typing import List, Sequence, Tuple, Union,Optional
import numpy as np
import cv2
import numbers
import torch
import onnxruntime
from collections import namedtuple
import random

def nms_numpy(boxes, scores, iou_threshold,score_threshold):
    """
    Pure numpy NMS implementation.
    :param boxes: [N, 4] where each box is represented as [x1, y1, x2, y2]
    :param scores: [N] where each score corresponds to a box
    :param iou_threshold: IoU threshold for NMS
    :return: indices of boxes to keep
    """
    keep = scores >= score_threshold
    boxes = boxes[keep]
    scores = scores[keep]
    
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]  # Sort boxes by score in descending order

    keep = []
    
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)

        intersection = w * h
        iou = intersection / (areas[i] + areas[order[1:]] - intersection)

        indices = np.where(iou <= iou_threshold)[0]
        order = order[indices + 1]

    return keep

cv2_interp_codes = {
    'nearest': cv2.INTER_NEAREST,
    'bilinear': cv2.INTER_LINEAR,
    'bicubic': cv2.INTER_CUBIC,
    'area': cv2.INTER_AREA,
    'lanczos': cv2.INTER_LANCZOS4 
}

class LoadImageFromFile():
    def transform(self, results: dict) -> dict:
        img = results['img']
        results['img_path'] = None
        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        return results


class YOLOv5KeepRatioResize():
    def __init__(self,scale: Union[int, Tuple[int, int]],keep_ratio: bool = True,**kwargs):
        assert keep_ratio is True
        self.keep_ratio = keep_ratio
        self.scale = scale
        
    def _get_rescale_ratio(self,old_size: Tuple[int, int],
                           scale: Union[float, Tuple[int]]) -> float:
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

    def _resize_img(self, results: dict):
        """Resize images with ``results['scale']``."""
        assert self.keep_ratio is True

        if results.get('img', None) is not None:
            image = results['img']
            original_h, original_w = image.shape[:2]
            ratio = self._get_rescale_ratio((original_h, original_w),self.scale)

            if ratio != 1:
                image = cv2.resize(
                    image,
                    (int(original_w * ratio), int(original_h * ratio)),
                    interpolation=cv2_interp_codes['area' if ratio < 1 else 'bilinear'])

            resized_h, resized_w = image.shape[:2]
            scale_ratio_h = resized_h / original_h
            scale_ratio_w = resized_w / original_w
            scale_factor = (scale_ratio_w, scale_ratio_h)

            results['img'] = image
            results['img_shape'] = image.shape[:2]
            results['scale_factor'] = scale_factor

    def transform(self, results: dict) -> dict:
        self._resize_img(results)
        return results


class LetterResize():
    def __init__(self,
                 scale: Union[int, Tuple[int, int]],
                 pad_val: dict = dict(img=0, mask=0, seg=255),
                 use_mini_pad: bool = False,
                 stretch_only: bool = False,
                 allow_scale_up: bool = True,
                 half_pad_param: bool = False,
                 **kwargs):
        self.scale = scale
        self.pad_val = pad_val
        self.use_mini_pad = use_mini_pad
        self.stretch_only = stretch_only
        self.allow_scale_up = allow_scale_up
        self.half_pad_param = half_pad_param

        if isinstance(pad_val, (int, float)):
            pad_val = dict(img=pad_val, seg=255)
        assert isinstance(
            pad_val, dict), f'pad_val must be dict, but got {type(pad_val)}'

        self.use_mini_pad = use_mini_pad
        self.stretch_only = stretch_only
        self.allow_scale_up = allow_scale_up
        self.half_pad_param = half_pad_param

    def impad(slef,img: np.ndarray,
          *,
          shape: Optional[Tuple[int, int]] = None,
          padding: Union[int, tuple, None] = None,
          pad_val: Union[float, List] = 0,
          padding_mode: str = 'constant') -> np.ndarray:

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

    def _resize_img(self, results: dict):
        """Resize images with ``results['scale']``."""
        image = results.get('img', None)
        if image is None:
            return

        # Use batch_shape if a batch_shape policy is configured
        if 'batch_shape' in results:
            scale = tuple(results['batch_shape'])  # hw
        else:
            scale = self.scale[::-1]  # wh -> hw

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
                image,(no_pad_shape[1], no_pad_shape[0]),
                interpolation=cv2.INTER_LINEAR)

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

            image = self.impad(
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
            results['pad_param'] = np.array(padding_list, dtype=np.float32)
        return results


    def transform(self, results: dict) -> dict:
        results = self._resize_img(results)
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


class Normalize():
    def __init__(self,mean,std):
        super().__init__()
        self.mean = torch.tensor(mean, dtype=torch.float32).reshape(1, 3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).reshape(1, 3, 1, 1)

    def forward(self, x):
        x = x[None].float()
        x -= self.mean.to(x.device)
        x /= self.std.to(x.device)
        return x

    

class ORTWrapper(torch.nn.Module):
    def __init__(self, weight,
                 device: Optional[torch.device]):
        super().__init__()

        if isinstance(device, str):
            device = torch.device(device)
        elif isinstance(device, int):
            device = torch.device(f'cuda:{device}')
        self.weight = weight
        self.device = device
        self.__init_session()
        self.__init_bindings()

    def __init_session(self):
        providers = ['CPUExecutionProvider']
        if 'cuda' in self.device.type:
            providers.insert(0, 'CUDAExecutionProvider')

        session = onnxruntime.InferenceSession(
            str(self.weight), providers=providers)
        self.session = session

    def __init_bindings(self):
        Binding = namedtuple('Binding', ('name', 'dtype', 'shape'))
        inputs_info = []
        outputs_info = []
        self.is_dynamic = False
        for i, tensor in enumerate(self.session.get_inputs()):
            if any(not isinstance(i, int) for i in tensor.shape):
                self.is_dynamic = True
            inputs_info.append(
                Binding(tensor.name, tensor.type, tuple(tensor.shape)))

        for i, tensor in enumerate(self.session.get_outputs()):
            outputs_info.append(
                Binding(tensor.name, tensor.type, tuple(tensor.shape)))
        self.inputs_info = inputs_info
        self.outputs_info = outputs_info
        self.num_inputs = len(inputs_info)

    def forward(self, *inputs):

        assert len(inputs) == self.num_inputs

        contiguous_inputs: List[np.ndarray] = [
            i.contiguous().cpu().numpy() for i in inputs
        ]

        if not self.is_dynamic:
            # make sure input shape is right for static input shape
            for i in range(self.num_inputs):
                assert contiguous_inputs[i].shape == self.inputs_info[i].shape

        outputs = self.session.run([o.name for o in self.outputs_info], {
            j.name: contiguous_inputs[i]
            for i, j in enumerate(self.inputs_info)
        })

        return tuple(torch.from_numpy(o).to(self.device) for o in outputs)
    
def get_image():
    step1 = LoadImageFromFile()
    step2 = YOLOv5KeepRatioResize((640,640))
    step3 = LetterResize((640,640),{'img':114},use_mini_pad=False,stretch_only=False,allow_scale_up=False,half_pad_param=False)
    step4 = Normalize([0.0,0.0,0.0],[255.0,255.0,255.0])
    bgr = cv2.imread(r'/home/apulis-test/userdata/code/mycode/my_mmyolo-main/mmyolo-main/data/cat/images/IMG_20210728_205117.jpg',1)
    # 转rgb
    rgb = cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
    image_info = dict(img=rgb,img_id=0)
    results = step1.transform(image_info)
    results = step2.transform(results)
    results = step3.transform(results)

    pad_param = results.get('pad_param',
                                np.array([0, 0, 0, 0], dtype=np.float32))
    h, w = results.get('ori_shape', rgb.shape[:2])
    pad_param = torch.tensor([pad_param[2], pad_param[0], pad_param[2], pad_param[0]],device="cuda:0")
    scale_factor = results.get('scale_factor', [1., 1])
    # scale_factor = torch.asarray(scale_factor * 2, device=args.device)
    scale_factor = torch.tensor(scale_factor * 2, device="cuda:0")
    image = torch.tensor(np.ascontiguousarray(np.transpose(results['img'], [2, 0, 1])),device="cuda:0")
    data = step4.forward(image)
    return data

    

if __name__=='__main__':
    step1 = LoadImageFromFile()
    step2 = YOLOv5KeepRatioResize((640,640))
    step3 = LetterResize((640,640),{'img':114},use_mini_pad=False,stretch_only=False,allow_scale_up=False,half_pad_param=False)
    step4 = Normalize([0.0,0.0,0.0],[255.0,255.0,255.0])
    bgr = cv2.imread(r'/home/apulis-test/userdata/code/mycode/my_mmyolo-main/mmyolo-main/data/cat/images/IMG_20210728_205117.jpg',1)
    # 转rgb
    rgb = cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
    image_info = dict(img=rgb,img_id=0)
    results = step1.transform(image_info)
    results = step2.transform(results)
    results = step3.transform(results)

    pad_param = results.get('pad_param',
                                np.array([0, 0, 0, 0], dtype=np.float32))
    h, w = results.get('ori_shape', rgb.shape[:2])
    pad_param = torch.tensor([pad_param[2], pad_param[0], pad_param[2], pad_param[0]],device="cuda:0")
    scale_factor = results.get('scale_factor', [1., 1])
    # scale_factor = torch.asarray(scale_factor * 2, device=args.device)
    scale_factor = torch.tensor(scale_factor * 2, device="cuda:0")
    image = torch.tensor(np.ascontiguousarray(np.transpose(results['img'], [2, 0, 1])),device="cuda:0")
    data = step4.forward(image)

    onnx_infer = ORTWrapper(weight=r'/home/apulis-test/userdata/code/mycode/my_mmyolo-main/mmyolo-main/work_dirs/yolov5_s-v61_fast_1xb12-40e_cat/epoch_40_0.65.onnx',\
                            device = 'cuda:0')

    # data_best = torch.load(r'/home/apulis-test/userdata/code/mycode/my_mmyolo-main/mmyolo-main/data.pth')
    # flag = torch.equal(data,data_best)
    output = onnx_infer(data)
    print(f'output:{output}')

    num_dets, bboxes, scores, labels = output
    scores = scores[0, :num_dets]
    bboxes = bboxes[0, :num_dets]
    labels = labels[0, :num_dets]
    bboxes -= pad_param
    bboxes /= scale_factor

    bboxes[:, 0::2].clamp_(0, w)
    bboxes[:, 1::2].clamp_(0, h)
    bboxes = bboxes.round().int()

    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(1000)]
    class_names = ('cat',)
    for (bbox, score, label) in zip(bboxes, scores, labels):
        bbox = bbox.tolist()
        color = colors[label]
        if class_names is not None:
            label_name = class_names[label]
            name = f'cls:{label_name}_score:{score:0.4f}'
        else:
            name = f'cls:{label}_score:{score:0.4f}'

        cv2.rectangle(bgr, bbox[:2], bbox[2:], color, 2)
        cv2.putText(
            bgr,
            name, (bbox[0], bbox[1] - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.0, [225, 255, 255],
            thickness=3)
        cv2.imwrite(r'/home/apulis-test/userdata/code/mycode/my_mmyolo-main/1.jpg',bgr)


    # 如果自己做nms
    onnx_infer2 = ORTWrapper(weight=r'/home/apulis-test/userdata/code/mycode/my_mmyolo-main/mmyolo-main/work_dirs/yolov5_s-v61_fast_1xb12-40e_cat/epoch_40_1_0.onnx',\
                        device = 'cuda:0')
    output2 = onnx_infer2(data)

    num_dets, bboxes, scores, labels = output2
    keep = nms_numpy(bboxes[0].cpu().clone().numpy(),scores[0].cpu().clone().numpy(),0.65,0.8)
    box = bboxes[0][keep]
    score = scores[0][keep]
    print(f'keep:{box}')
        







