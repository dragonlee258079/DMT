# DMT
Code release for the CVPR 2023 paper `"Discriminative Co-Saliency and Background Mining Transformer for Co-Salient Object Detection"`.

![avatar](framework.jpg)

## Abstract
Most previous co-salient object detection works mainly focus on extracting co-salient cues via mining the consistency relations across images while ignore **explicit** exploration of background regions. In this paper, we propose a Discriminative co-saliency and background Mining Transformer framework (DMT) based on several economical multi-grained correlation modules to **explicitly** mine both co-saliency and background information and effectively model their discrimination. Specifically, we first propose a region-to-region correlation module for introducing inter-image relations to pixel-wise segmentation features while maintaining computational efficiency. Then, we use two types of pre-defined tokens to mine co-saliency and background information via our proposed contrast-induced pixel-to-token correlation and co-saliency token-to-token correlation modules. We also design a token-guided feature refinement module to enhance the discriminability of the segmentation features under the guidance of the learned tokens. We perform iterative mutual promotion for the segmentation feature extraction and token construction. Experimental results on three benchmark datasets demonstrate the effectiveness of our proposed method. 

## Result
The prediction results of our dataset can be download from [prediction](https://pan.baidu.com/s/1erKtadxG8NJoCMeW6fuofQ) (jjht).

![alt_text](./result.jpg)

## Environment Configuration
- Linux with Python ≥ 3.6
- PyTorch ≥ 1.7 and [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
  Install them together at [pytorch.org](https://pytorch.org) to make sure of this. Note, please check
  PyTorch version matches that is required by Detectron2.
- Detectron2: follow [Detectron2 installation instructions](https://detectron2.readthedocs.io/tutorials/install.html).
- OpenCV is optional but needed by demo and visualization
- `pip install -r requirements.txt`

## Data Preparation
````
-- dataset
   |-- train_data
   |   |-- | CoCo9k
   |   |-- |-- | img
   |   |-- |-- | gt
   |   |-- | DUTS_class
   |   |-- |-- | img
   |   |-- |-- | gt
   |   |-- | DUTS_class_syn
   |   |-- |-- | img_png_seamless_cloning_add_naive
   |   |-- |-- |-- | img
   |   |-- |-- |-- | co_gt
   |   |-- |-- | img_png_seamless_cloning_add_naive_reverse_2
   |   |-- |-- |-- | img
   |   |-- |-- |-- | co_gt
   |-- test_data
   |   |-- | CoCA
   |   |-- | CoSal2015
   |   |-- | CoSOD3k
   |   |-- |-- | rgb
   |   |-- |-- |-- | COCO_train2014_000000000110.jpg ...
   |   |-- |-- | gt
   |   |-- |-- |-- | COCO_train2014_000000000110.png ...
   |   |-- test
   |   |-- |-- | rgb
   |   |-- |-- |-- | COCO_val2014_000000000192.jpg ...
   |   |-- |-- | gt
   |   |-- |-- |-- | COCO_val2014_000000000192.png ...
````
