from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import cv2
import numpy as np
import  os

crowdpose_val_annotations_path = '/data/posework/crowdpose/json/crowdpose_val.json'

coco = COCO(crowdpose_val_annotations_path)
img_ids = coco.getImgIds()
ann_ids = coco.getAnnIds(img_ids[0])

img_info = coco.loadImgs(img_ids[0])
print(img_info)

anns = coco.loadAnns(ann_ids)
print(anns[0])

image_path = os.path.join('/data/posework/crowdpose/images', img_info[0]['file_name'])
img = cv2.imread(image_path)

kk = './kk'
keypoints = np.array(anns[0]['keypoints']).reshape(14,3)
for i in range(14):
    if keypoints[i][2]>0 :
        #pdb.set_trace()
        cv2.circle(img,tuple(keypoints[i][:2].astype(np.int32)),10,(255,0,255), 10)
        save_name = os.path.join(kk, '{}.jpg'.format(i))
        cv2.imwrite(save_name, img)
