# 函数:make_dataset
# 将多个数据集合并成一个数据集
# 首先遍历每个文件夹下是否有labelme文件夹,如果没有,生成labelme;如果有,下一步;
# labelme是否在目的文件中，如果不在，复制过去
# labelmetoccoco
import os
import shutil
from pycocotools.coco import COCO
from tqdm import tqdm
import json
import datetime
import cv2
import numpy as np

def walk_file(path, type=".json"):
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(type):
                f_path = os.path.join(root, file)
                if os.path.exists(f_path) and os.path.exists(f_path.replace('.json', '.jpg')):
                    yield f_path

def coco_to_labelme(path):
    if os.path.exists(os.path.join(path,'labelme')):
        # 删除文件夹及其中所有内容
        shutil.rmtree(os.path.join(path,'labelme'))
        print(f"文件夹 {os.path.join(path,'labelme')} 及其内容已删除")
    os.makedirs(os.path.join(path,"labelme"), exist_ok=True)
    coco = COCO(os.path.join(path,"annotations","instances_annotations.json"))
    categories = {cat['id']: cat['name'] for cat in coco.loadCats(coco.getCatIds())}
    image_ids = coco.getImgIds()
    no_image = []
    for img_id in tqdm(image_ids):
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(path,'images', img_info['file_name'])
        if not os.path.exists(img_path):
            no_image.append(img_path)
            continue
        ann_ids = coco.getAnnIds(imgIds=img_id)
        annotations = coco.loadAnns(ann_ids)
        shapes = []
        lable_name = []
        for ann in annotations:
            lable_name.append(categories[ann['category_id']])
            bbox = ann['bbox']
            shape = {
                'label': categories[ann['category_id']],
                'points': [
                    [bbox[0], bbox[1]],
                    [bbox[0] + bbox[2], bbox[1] + bbox[3]]
                ],
                'group_id': None,
                "description": "",
                'shape_type': 'rectangle',
                'flags': {}
            }
            shapes.append(shape)
        ann_json = {
                "version": "5.3.1",
                "flags": {},
                "shapes": shapes,
                "imagePath": img_info['file_name'],
                "imageData": None,
                "imageHeight": img_info['height'],
                "imageWidth": img_info['width']
        }

        image_path = os.path.join(path,'images',img_info['file_name'])
        json_save_path = os.path.join(path,'labelme', img_info['file_name'].replace('.jpg', '.json'))
        if not os.path.exists(json_save_path):
            with open(json_save_path, 'w',encoding='utf-8') as f:
                json.dump(ann_json, f, indent=4)
            f.close()
            imagepath = json_save_path.replace('.json','.jpg')
            shutil.copy2(img_path, imagepath)
        else:
            new_json_save_path = json_save_path[0:-5]+'_'+str(img_id)+'.json'
            with open(new_json_save_path, 'w',encoding='utf-8') as f:
                json.dump(ann_json, f, indent=4)
            f.close()
            new_imagepath = new_json_save_path.replace('.json','.jpg')
            shutil.copy2(img_path, new_imagepath)


def labelme_to_coco(path,label_name):
    startIndex = 1
    if os.path.exists(os.path.join(path,'annotations')):
        shutil.rmtree(os.path.join(path,'annotations'))
        print(f"文件夹 {os.path.join(path,'annotations')} 及其内容已删除")
    os.makedirs(os.path.join(path,"annotations"), exist_ok=True)

    now = datetime.datetime.now()

    data = dict(
        licenses=[
            {
                "url":"",
                "id":0,
                "name":""
            }
        ],
        info={
            "contributor": "",
            "date_created": "",
            "description":"",
            "url":"",
            "version":"",
            "year":""
        },
        categories=[
            # supercategory, id, name
        ],

        images=[
            # license, url, file_name, height, width, date_captured, id
        ],
        annotations=[
            # segmentation, area, iscrowd, image_id, bbox, category_id, id
        ],
    )

    class_name_to_id = {}

    for i, line in enumerate(label_name):
        class_id = i  # starts with -1
        class_name = line
        class_name_to_id[class_name] = class_id
        data["categories"].append(
            {
                "id":class_id+startIndex,  # id从1开始
                "name":class_name,
                "supercategory":""
            }
        )

    out_ann_file = os.path.join(os.path.join(path,'annotations'), "instances_annotations.json")

    label_files = list(walk_file(path))

    for image_id, filename in enumerate(tqdm(label_files)):
        assert os.path.exists(filename)
        # print(filename)
        label_file = json.load(open(filename, 'r',encoding='UTF-8'))   #labelme.LabelFile(filename=filename)

        base = os.path.splitext(os.path.basename(filename))[0]
        out_file_name = base + ".jpg"

        img_path = filename.replace(".json", ".jpg")
        assert os.path.exists(img_path), img_path
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        data["images"].append(
            {
                "id":image_id,  # 图像id从1开始
                "width":img.shape[1],
                "height":img.shape[0],
                "file_name":out_file_name,
                "license":0,
                "flickr_url":"",
                "coco_url" :"",
                "date_captured":0,
            }
        )

        Area = []
        Bbox = []
        Labelid = []
        for shape in label_file['shapes']:
            points = shape["points"]
            label = shape["label"]
            labelid = label_name.index(label)+startIndex
            group_id = shape["group_id"]
            shape_type = shape["shape_type"]

            xmin = round(points[0][0],2)
            ymin = round(points[0][1],2)
            xmax = round(points[1][0],2)
            ymax = round(points[1][1],2)

            o_width = round(xmax-xmin,2)
            o_height = round(ymax - ymin,2)
            seg_area = o_width*o_height

            Labelid.append(labelid)
            Area.append(seg_area)
            Bbox.append([xmin,ymin,o_width,o_height])

        for area, bbox,labelid in zip(Area,Bbox,Labelid):
            data["annotations"].append(
                {
                    "id":len(data["annotations"]),
                    "image_id":image_id,
                    "category_id":labelid,
                    "segmentation":[],
                    "area":area,
                    "bbox":bbox,
                    "iscrowd":0,
                    "attributes": {"occluded": False}
                }
            )
    with open(out_ann_file, "w",encoding='UTF-8') as f:
        json.dump(data, f)


def is_illega_labelme(path):
    images = [name for name in os.listdir(os.path.join(path,"labelme")) if not name.endswith('.json')]
    jsons = [name.replace('.json','.jpg') for name in os.listdir(os.path.join(path,"labelme")) if name.endswith('.json')]
    up_images = os.listdir(os.path.join(path,'images'))

    if len(os.listdir(os.path.join(path,"labelme")))==2*len(os.listdir(os.path.join(path,"images"))):
        if sorted(images) == sorted(jsons):
            if sorted(images) == sorted(up_images):
                result = '合格'
            else:
                result = '不合格'
        else:
            result = '不合格'
    else:
        result = '不合格'
    return result



if __name__ == '__main__':
    labelmeName = ['侧立', '偏移', '反向', '反白', '少件', '少锡', '异物', '引脚错位', '损件', '短路', '立碑', '翘脚', '错件', '锡珠', '虚焊']
    dataset_path = [r'/home/apulis-test/teamdata/jlc_dataset/jingshi-project/jlc_base_dataset/',
                    r'/home/apulis-test/teamdata/jlc_dataset/jingshi-project/jingshi_datasets_0110/',
                    r'/home/apulis-test/teamdata/jlc_dataset/jingshi-project/jingshi_datasets_0112/',
                    r'/home/apulis-test/teamdata/jlc_dataset/jingshi-project/jingshi_dataset_0113/',
                    r'/home/apulis-test/teamdata/jlc_dataset/jingshi-project/jingshi_dataset_0114/',
                    r'/home/apulis-test/teamdata/jlc_dataset/jingshi-project/jingshi_dataset_0116/',
                    r'/home/apulis-test/teamdata/jlc_dataset/jingshi-project/jingshi_dataset_0121/',
                    r'/home/apulis-test/teamdata/jlc_dataset/jingshi-project/jingshi_dataset_0122/',
                    r'/home/apulis-test/teamdata/jlc_dataset/jingshi-project/jingshi_dataset_0123/',
                    r'/home/apulis-test/teamdata/jlc_dataset/jingshi-project/jingshi_dataset_0206/',
                    r'/home/apulis-test/teamdata/jlc_dataset/jingshi-project/jingshi_dataset_0207/',
                    r'/home/apulis-test/teamdata/jlc_dataset/jingshi-project/jingshi_dataset_0214/',
                    r'/home/apulis-test/teamdata/jlc_dataset/jingshi-project/jingshi_dataset_0221',
                    r'/home/apulis-test/teamdata/jlc_dataset/jingshi-project/jingshi_dataset_0302',
                    r'/home/apulis-test/teamdata/jlc_dataset/jingshi-project/jingshi_dataset_0307',
                    r'/home/apulis-test/teamdata/jlc_dataset/jingshi-project/jingshi_dataset_0314',
                    r'/home/apulis-test/teamdata/jlc_dataset/jingshi-project/jingshi_dataset_0318',
                    r'/home/apulis-test/teamdata/jlc_dataset/jingshi-project/jingshi_dataset_0321',
                   ]
    
    concat_dataset_path = r'/home/apulis-test/teamdata/jlc_dataset/jingshi-project/concat_dataset'

    # 生成labelme
    for sub_path in dataset_path:
        if not os.path.exists(os.path.join(sub_path,"labelme")):
            coco_to_labelme(sub_path)
            if is_illega_labelme(sub_path) == '合格':
                print(f"dataset_path:{sub_path}:生成labelme文件夹")
        else:
            # 检查labelme是否合法
            result = is_illega_labelme(sub_path)
            if result == '不合格':
                coco_to_labelme(sub_path)
                print(f"dataset_path:{sub_path}:生成labelme文件夹")
            else:
                print(f"dataset_path:{sub_path}:labelme文件夹未变化")

    # 合并labelme
    if not os.path.exists(os.path.join(concat_dataset_path,"labelme")):
        os.makedirs(os.path.join(concat_dataset_path,"labelme"), exist_ok=True)
    if not os.path.exists(os.path.join(concat_dataset_path,"images")):
        os.makedirs(os.path.join(concat_dataset_path,"images"), exist_ok=True)
    if not os.path.exists(os.path.join(concat_dataset_path,"annotations")):
        os.makedirs(os.path.join(concat_dataset_path,"annotations"), exist_ok=True)

    images = [name for name in os.listdir(os.path.join(concat_dataset_path,"labelme")) if not name.endswith('.json')]
    jsons = [name for name in os.listdir(os.path.join(concat_dataset_path,"labelme")) if name.endswith('.json')]

    for sub_path in tqdm(dataset_path):
        sub_images = [name for name in os.listdir(os.path.join(sub_path,"labelme")) if not name.endswith('.json')]
        sub_jsons = [name for name in os.listdir(os.path.join(sub_path,"labelme")) if name.endswith('.json')]
        for simage in tqdm(sub_images):
            if simage not in images:
                shutil.copy2(os.path.join(sub_path,"labelme",simage),os.path.join(concat_dataset_path,"labelme",simage))
                shutil.copy2(os.path.join(sub_path,"labelme",simage.replace('.jpg','.json')),os.path.join(concat_dataset_path,"labelme",simage.replace('.jpg','.json')))
                shutil.copy2(os.path.join(sub_path,"labelme",simage),os.path.join(concat_dataset_path,"images",simage))

    # 检查labelme
    result = is_illega_labelme(concat_dataset_path)
    if result == '合格':
        print("合并数据集结果合格OK")
        print("~~~~~~~~~~~~~~~~~~")
        labelme_to_coco(concat_dataset_path,labelmeName)
        num = len(os.listdir(os.path.join(concat_dataset_path,"images")))
        print(f"数据集图片总数为:{num}")
        print('完成')
    else:
        print("合并数据集结果不不不合格")
        exit()




    

    


            
