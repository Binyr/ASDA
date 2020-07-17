## Environment
python 3.7
torch==1.0.1.post2
torchvision==0.2.2

EasyDict==1.7
opencv-python==3.4.1.15
shapely==1.6.4
Cython
scipy
pandas
pyyaml
json_tricks
scikit-image
yacs>=0.1.5
tensorboardX==1.6

## Quick start
1. install dependency

2. Make libs
```
cd ${POSE_ROOT}/lib
make
```

3. Install COCOAPI:
```
# COCOAPI=/path/to/clone/cocoapi
git clone https://github.com/cocodataset/cocoapi.git $COCOAPI
cd $COCOAPI/PythonAPI
# Install into global site-packages
make install
# Alternatively, if you do not have permissions or prefer
# not to install the COCO API into global site-packages
python3 setup.py install --user
```
Note that instructions like # COCOAPI=/path/to/install/cocoapi indicate that you should pick a path where you'd like to have the software cloned and then set an environment variable (COCOAPI in this case) accordingly.

4. Init output(training model output directory) and log(tensorboard log directory) directory:
```
mkdir output 
mkdir log
```

   Your directory tree should look like this:
```
${POSE_ROOT}
├── data
├── experiments
├── lib
├── log
├── models
├── output
├── tools 
├── README.md
└── requirements.txt
```
5. Download pretrained models from HRNet model zoo([GoogleDrive](https://drive.google.com/drive/folders/1hOTihvbyIxsm5ygDpbUuJ7O_tzv4oXjC?usp=sharing) or [OneDrive](https://1drv.ms/f/s!AhIXJn_J-blW231MH2krnmLq5kkQ))
```
${POSE_ROOT}
 |-- models
     |-- pytorch
         |-- imagenet
         |   |-- hrnet_w32-36af842e.pth
         |   |-- hrnet_w48-8ef0771d.pth
         |   |-- resnet50-19c8e357.pth
         |   |-- resnet101-5d3b4d8f.pth
         |   |-- resnet152-b121ed2d.pth
         |-- pose_coco
         |   |-- pose_hrnet_w32_256x192.pth
         |   |-- pose_hrnet_w32_384x288.pth
         |   |-- pose_hrnet_w48_256x192.pth
         |   |-- pose_hrnet_w48_384x288.pth
         |   |-- pose_resnet_101_256x192.pth
         |   |-- pose_resnet_101_384x288.pth
         |   |-- pose_resnet_152_256x192.pth
         |   |-- pose_resnet_152_384x288.pth
         |   |-- pose_resnet_50_256x192.pth
         |   |-- pose_resnet_50_384x288.pth
         |-- pose_mpii
             |-- pose_hrnet_w32_256x256.pth
             |-- pose_hrnet_w48_256x256.pth
             |-- pose_resnet_101_256x256.pth
             |-- pose_resnet_152_256x256.pth
             |-- pose_resnet_50_256x256.pth
```
### Data preparation
**For MPII data**, please download from [MPII Human Pose Dataset](http://human-pose.mpi-inf.mpg.de/). The original annotation files are in matlab format. We use the json format from HRNet. You can download them from [OneDrive](https://1drv.ms/f/s!AhIXJn_J-blW00SqrairNetmeVu4) or [GoogleDrive](https://drive.google.com/drive/folders/1En_VqmStnsXMdldXA6qpqEyDQulnmS3a?usp=sharing).
Extract them under {POSE_ROOT}/data, and make them look like this:
```
${POSE_ROOT}
|-- data
`-- |-- mpii
    `-- |-- annot
        |   |-- gt_valid.mat
        |   |-- test.json
        |   |-- train.json
        |   |-- trainval.json
        |   `-- valid.json
        `-- images
            |-- 000001163.jpg
            |-- 000003072.jpg
```

**For COCO data**, please download from [COCO download](http://cocodataset.org/#download), 2017 Train/Val is needed for COCO keypoints training and validation. We use the person detection result provided by HRNet to reproduce our multi-person pose estimation results. You could download from [OneDrive](https://1drv.ms/f/s!AhIXJn_J-blWzzDXoz5BeFl8sWM-) or [GoogleDrive](https://drive.google.com/drive/folders/1fRUDNUDxe9fjqcRZ2bnF_TKMlO0nB_dk?usp=sharing).
Download and extract them under {POSE_ROOT}/data, and make them look like this:
```
${POSE_ROOT}
|-- data
`-- |-- coco
    `-- |-- annotations
        |   |-- person_keypoints_train2017.json
        |   `-- person_keypoints_val2017.json
        |-- person_detection_results
        |   |-- COCO_val2017_detections_AP_H_56_person.json
        |   |-- COCO_test-dev2017_detections_AP_H_609_person.json
        `-- images
            |-- train2017
            |   |-- 000000000009.jpg
            |   |-- 000000000025.jpg
            |   |-- 000000000030.jpg
            |   |-- ... 
            `-- val2017
                |-- 000000000139.jpg
                |-- 000000000285.jpg
                |-- 000000000632.jpg
                |-- ... 
```

### Training and Testing
#### Training on MPII dataset

```
python tools/train_stn.py \
    --cfg experiments/mpii/hrnet/stn/w32_256x256_adam_lr1e-3_adversalstnlr0001_posefirst_res18_numPart1_bmp.yaml
```

#### Testing on MPII dataset 

```
python tools/test.py \
--cfg experiments/mpii/hrnet/stn/w32_256x256_adam_lr1e-3_adversalstnlr0001_posefirst_res18_numPart1_bmp.yaml \
TEST.MODEL_FILE path/to/res_dir/model_best.pth
```

#### Multi-scale voting test
```
python tools/test_multiscale_multistage_voting.py \
--cfg experiments/mpii/hrnet/stn/w32_256x256_adam_lr1e-3_adversalstnlr0001_posefirst_res18_numPart1_bmp.yaml \
TEST.MODEL_FILE path/to/res_dir/model_best.pth
```
可在文件中调整参数'num_stage_to_fuse' 和 'scale_pyramid'来融合不同的尺度和阶段的预测。为了避免反复预测，试运行该文件后，将在${POSE_ROOT}下保存一个名为'test_preds_pyramid_mstage.npy'的文件，该文件中存储了num_scale个尺度和num_stage个阶段的预测结果。**当需要对不同的模型多尺度预测时，需要预先删除文件'test_preds_pyramid_mstage.npy'。**

#### Training on COCO dataset

```
python tools/train_stn.py \
    --cfg experiments/coco/hrnet/w32_256x192_adam_lr1e-3_numPart1_bmp_notBothHfpSa.yaml
```

#### Testing on COCO dataset 

```
python tools/test.py \
--cfg experiments/coco/hrnet/w32_256x192_adam_lr1e-3_numPart1_bmp_notBothHfpSa.yaml \
TEST.MODEL_FILE path/to/res_dir/model_best.pth
```

#### Visualizing augmented image
```
python tools/visualize_stn.py \
--cfg experiments/mpii/hrnet/stn/w32_256x256_adam_lr1e-3_adversalstnlr0001_posefirst_res18_numPart1_bmp.yaml
```
可视化结果将存储与'path/to/output_dir/stn_vis/'

#### Important Parameters
| Name        | Type  | Optimal | Function |       
| --------    | -----  | ----  |--------|
| ASA.NUM_AUG     | tuple   |(1,)| 贴图数量 |
| ASA.PART_ANN_FILE | str  |'./lip/parts_bmp_filter_done/part_anns.json'|部件标注文件|
| ASA.PART_ROOT_DIR | str  | './lip/parts_bmp_filter_done/' |部件文件夹|
| ASA.ERODE_KERNEL|tuple|(3,3)| 用于模糊部件边界|
|ASA.GAUSSIAN_KERNEL|tuple|(3,3)|用于模糊部件边界|
|ASA.BOTH_HF_SA|bool|False|控制是否同时施加'半身人增强'和'贴图增强'。注意MPII数据集未开启'半身人增强'|
|STN.LR|float|0.001|生成网络学习率|
|STN.STN_FIRST| bool | 0 | 控制先生成器 |
|STN.NG|int|1|每轮生成器迭代次数|
|STN.ND|int|1|每轮判别器迭代次数|