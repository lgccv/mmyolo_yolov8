# 基础模型下载地址 no mask refine
https://github.com/open-mmlab/mmyolo/tree/main/configs/yolov8
yolov8_l_syncbn_fast_8xb16-500e_coco_20230217_182526-189611b6.pth

通过百度网盘分享的文件：yolov8_l_syncbn_fast_8xb16-500e_coc...
网盘链接：https://pan.baidu.com/s/1SI8HWNuIc3hHh-XUYk0ohQ?pwd=x3fh 
提取码：x3fh 


# 精视项目
python make_dataset/make_wafer_defect_dataset.py &&  CUDA_VISIBLE_DEVICES=0,1 ./tools/dist_train.sh configs/AD_mlops/AD_wafer_defect.py 2
CUDA_VISIBLE_DEVICES=0,1 ./tools/dist_train.sh configs/AD_mlops/AD_wafer_defect.py 2
python projects/easydeploy/tools/export_onnx.py configs/AD_mlops/AD_wafer_defect.py work_dirs/AD_wafer_defect/epoch_200.pth --work-dir work_dirs/AD_jingshi --img-size 640 640 --simplify
python projects/easydeploy/tools/export_onnx.py configs/AD_mlops/AD_wafer_defect.py work_dirs/AD_wafer_defect/epoch_200.pth --work-dir work_dirs/AD_jingshi --img-size 640 640 --simplify --fp16-export


# 环境
python 3.8
addict                 2.4.0
albumentations         1.4.3
aliyun-python-sdk-core 2.15.0
aliyun-python-sdk-kms  2.16.2
asttokens              2.4.1
backcall               0.2.0
certifi                2024.2.2
cffi                   1.16.0
charset-normalizer     3.3.2
click                  8.1.7
colorama               0.4.6
coloredlogs            15.0.1
comm                   0.2.2
contourpy              1.1.1
crcmod                 1.7
cryptography           42.0.5
cycler                 0.12.1
debugpy                1.6.7
decorator              5.1.1
executing              2.0.1
flatbuffers            24.3.25
fonttools              4.51.0
humanfriendly          10.0
idna                   3.6
imageio                2.34.0
importlib_metadata     7.1.0
importlib_resources    6.4.0
ipykernel              6.29.3
ipython                8.12.2
jedi                   0.19.1
jmespath               0.10.0
joblib                 1.3.2
jupyter_client         8.6.1
jupyter_core           5.5.0
kiwisolver             1.4.5
lazy_loader            0.4
Markdown               3.6
markdown-it-py         3.0.0
matplotlib             3.7.5
matplotlib-inline      0.1.6
mdurl                  0.1.2
mkl-fft                1.3.8
mkl-random             1.2.4
mkl-service            2.4.0
mmcv                   2.0.1
mmdet                  3.3.0
mmengine               0.10.3
model-index            0.1.11
mpmath                 1.3.0
nest-asyncio           1.6.0
netron                 7.8.4
networkx               3.1
numpy                  1.24.4
onnx                   1.16.2
onnxruntime            1.19.0
opencv-python          4.9.0.80
opencv-python-headless 4.9.0.80
opendatalab            0.0.10
openmim                0.3.9
openxlab               0.0.37
ordered-set            4.1.0
oss2                   2.17.0
packaging              24.0
pandas                 2.0.3
parso                  0.8.4
pexpect                4.9.0
pickleshare            0.7.5
pillow                 10.2.0
pip                    23.3.1
platformdirs           4.2.0
prettytable            3.10.0
prompt-toolkit         3.0.43
protobuf               5.27.3
psutil                 5.9.0
ptyprocess             0.7.0
pure-eval              0.2.2
pycocotools            2.0.7
pycparser              2.22
pycryptodome           3.20.0
Pygments               2.17.2
pyparsing              3.1.2
python-dateutil        2.9.0
pytz                   2023.4
PyWavelets             1.4.1
PyYAML                 6.0.1
pyzmq                  25.1.2
requests               2.28.2
rich                   13.4.2
scikit-image           0.21.0
scikit-learn           1.3.2
scipy                  1.10.1
setuptools             60.2.0
shapely                2.0.3
six                    1.16.0
stack-data             0.6.2
sympy                  1.13.2
tabulate               0.9.0
termcolor              2.4.0
terminaltables         3.1.10
threadpoolctl          3.4.0
tifffile               2023.7.10
timm                   0.4.12
tomli                  2.0.1
torch                  1.10.1
torchvision            0.11.2
tornado                6.3.3
tqdm                   4.65.2
traitlets              5.14.2
trtpy                  1.2.6
typing_extensions      4.9.0
tzdata                 2024.1
urllib3                1.26.18
wcwidth                0.2.13
wheel                  0.41.2
yacs                   0.1.8
yapf                   0.40.2
zipp                   3.17.0

