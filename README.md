# **Y**ou **O**nly **L**ook **A**t **C**oefficien**T**s
```
    ██╗   ██╗ ██████╗ ██╗      █████╗  ██████╗████████╗
    ╚██╗ ██╔╝██╔═══██╗██║     ██╔══██╗██╔════╝╚══██╔══╝
     ╚████╔╝ ██║   ██║██║     ███████║██║        ██║   
      ╚██╔╝  ██║   ██║██║     ██╔══██║██║        ██║   
       ██║   ╚██████╔╝███████╗██║  ██║╚██████╗   ██║   
       ╚═╝    ╚═════╝ ╚══════╝╚═╝  ╚═╝ ╚═════╝   ╚═╝ 
```
# Train-Yolact-on-custom-dataset
YOLACT++ is an advanced real-time instance segmentation model designed to accurately identify objects within images while also segmenting them into individual instances. Unlike traditional object detection systems, which typically rely on bounding boxes, YOLACT++ provides pixel-level segmentation for each object instance. This is achieved through a combination of techniques, including a fully convolutional network architecture and a prototype mask generation mechanism. By accurately delineating object boundaries and providing real-time performance, YOLACT++ is well-suited for applications such as autonomous driving, surveillance, and robotics, where both speed and precision are crucial.

![Example 2](data/yolact_example_2.png)

# Installation
 - Clone this repository and enter it:
   ```Shell
   git clone https://github.com/dbolya/yolact.git
   cd yolact
   ```
 - Set up the environment using one of the following methods:
   - Using [Anaconda](https://www.anaconda.com/distribution/)
     - Run `conda env create -f environment.yml`
   - Manually with pip
     - Set up a Python3 environment (e.g., using virtenv).
     - Install [Pytorch](http://pytorch.org/) 1.0.1 (or higher) and TorchVision.
     - Install some other packages:
       ```Shell
       # Cython needs to be installed before pycocotools
       pip install cython
       pip install opencv-python pillow pycocotools matplotlib 

# Dataset folder structure


- Dataset folder
    - Train (images with annotations)
    - Test (images with annotations)


## Dataset formatting
```Shell
# Run just the raw model on the first 1k images of the validation set
python3 labelme2coco.py {train dataset folder path} {Train output folder path} --label=labels.txt
```
same thing should be done for test folder also , make sure you have added your labels in labels.txt

## config file changes

In data/config.py below changes should be done :
- In Dataset section add below thing at the end.
```Shell
my_dataset = dataset_base.copy({
  'name': 'coin dataset',
  'train_info': '{you train annotation.json path}',
  'train_images': '{training images path}',
  'valid_info': '{you test annotation.json path}',
  'valid_images': '{testing images path}',
  'class_names': ('your_label',),
  'label_map': { 0:  1 }  # if the labels are more than then simulataneously label_map should be vary like {0:1,1:2,2:3 and so on}
})
```

- In Yolact V1.0 section add below thing at the end.
```Shell
my_resnet50_config = yolact_resnet50_config.copy({
    'name': 'yolact_plus_resnet50_yokohama',
    # Dataset stuff
    'dataset': my_dataset,
    'num_classes': len(my_dataset.class_names) + 1,
    # Image Size
    'max_size': 512,
})
```

- To train, grab an imagenet-pretrained model and put it in `./weights`.
  - For Resnet101, download `resnet101_reducedfc.pth` from [here](https://drive.google.com/file/d/1tvqFPd4bJtakOlmn-uIA492g2qurRChj/view?usp=sharing).
  - For Resnet50, download `resnet50-19c8e357.pth` from [here](https://drive.google.com/file/d/1Jy3yCdbatgXa5YYIdTCRrSV0S9V5g1rn/view?usp=sharing).
  - For Darknet53, download `darknet53.pth` from [here](https://drive.google.com/file/d/17Y431j4sagFpSReuPNoFcj9h7azDTZFf/view?usp=sharing).

# Training
```Shell
python3 ./train.py --config=my_resnet50_config #change config as per mentioned in config file for custom dataset
```

# Evaluation
```Shell
python3  eval.py --trained_model={path to your trained model}  --config=my_resnet50_config  --score_threshold=0.50 --images={testing image path}:{tested output images path}
```

Reference : https://github.com/dbolya/yolact
