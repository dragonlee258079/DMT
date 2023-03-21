# DMT
Code release for the CVPR 2023 paper `"Discriminative Co-Saliency and Background Mining Transformer for Co-Salient Object Detection"`.

![avatar](framework.jpg)

## Abstract
Most previous co-salient object detection works mainly focus on extracting co-salient cues via mining the consistency relations across images while ignore **explicit** exploration of background regions. In this paper, we propose a Discriminative co-saliency and background Mining Transformer framework (DMT) based on several economical multi-grained correlation modules to **explicitly** mine both co-saliency and background information and effectively model their discrimination. Specifically, we first propose a region-to-region correlation module for introducing inter-image relations to pixel-wise segmentation features while maintaining computational efficiency. Then, we use two types of pre-defined tokens to mine co-saliency and background information via our proposed contrast-induced pixel-to-token correlation and co-saliency token-to-token correlation modules. We also design a token-guided feature refinement module to enhance the discriminability of the segmentation features under the guidance of the learned tokens. We perform iterative mutual promotion for the segmentation feature extraction and token construction. Experimental results on three benchmark datasets demonstrate the effectiveness of our proposed method. 

## Result
The prediction results of our dataset can be download from [prediction](https://pan.baidu.com/s/1erKtadxG8NJoCMeW6fuofQ) (jjht).

![alt_text](./result.jpg)
