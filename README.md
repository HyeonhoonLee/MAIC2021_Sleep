# MAIC2021_Sleep

SNUH Medical AI Challenge 2021

Sleep AI challenge

SNUH x OUAR LAB X MBRC X NIA X MNC.AI

Teamname : SleepingDragon 

Crews: Hyeonhoon Lee, Si Young Yie, MinSeok Hong, SeungHoon Lee


1. File flow
  
1) 파일불러오기: loaderViTrgb.py 를 통해 데이터를 불러오게됩니다.
2) 모델 학습: 두 가지를 각각 돌려야합니다.
  - 모델1(Vision Transformer): mainViTrgb.py 실행
  - 모델2(Efficientnet b4): mainEff.py 실행
3) 모델 학습 시 epoch 별로 weight가 저장이 되는데, 저희 팀에서는 loss와 metric을 고려하여 모델1은 epoch4, 모델2는 epoch1을 사용하였으며 그 이름은 아래와 같습니다.
  - 모델1 weight: vit_base_patch16_224_fold_0_4
  - 모델2 weight: tf_efficientnet_b4_ns_fold_0_1
4) Inference: 모델 1,2의 가중치의 각각에 대해 각 모델에서 추론이 될 수 있도록 한 파일입니다.: inferblend.py 실행
5) 모델 추론 시 바로 prediction label을 저장하는게 아니라 향후 앙상블을 위해 prediction 확률 값을 저장(.npy 의 numpy array)합니다.
  - 모델1에서 나온 각 클래스별 확률 numpy array: vit_base_patch16_224_fold_0_[4].npy
  - 모델2에서 나온 각 클래스별 확률 numpy array: tf_efficientnet_b4_ns_fold_0_[1].npy
6) 두 행렬을 모두 이용하여 앙상블을 진행합니다: blend.py 실행
7) blend.py가 실행되면 filesblend_sqrt_[4]_[1].csv 라는 prediction 결과(후처리 전)가 생성됩니다.
8) 마지막으로 postprocess.py를 실행하면 최종 결과물인 files_post.csv가 나오게됩니다. 이 결과를 최종제출하였습니다.



2. Packages list

- numpy, pandas 
- sklearn, pytorch
- timm, torchvision
- cv2, matplotlib, PIL, Fmix, scipy, adamp, albumentations
- datetime, glob, time, math,  random 
- skimage, os, multiprocessing, tqdm, sys

3. Data preprocessing
- We tried image segmentation by each signals and trained the model with some meaningful segments,
  but the performance was not good. 
- Data imbalance was identified. But we thought that the degree of imbalance was not much severe. 
- Although upsampling might help the better performance of model, we decided not to do that.
- Because, the number of data is too large for the given computing environment. It takes much time to train 1 epoch (about 1.5hr to 2.0hr)
- However, some augmentation methods were used as follows: 
    1) Fmix (a variant of cutMix and MixUp) by recent published article (https://arxiv.org/abs/2002.12047)
    2) Coarsedropout and cutout by the package called albumentations

 
4. Modelling

- Model Architecture: Vision Transformer(ViT), Efficientnet-b4
- Tool: python, Pytorch
- Ensemble method: Use the weighted average of the square root of probability of each classes in models

5. Training
- Loss function: categorical cross entropy with label smoothing
- Training method: Automatic Mixed Precision(AMP)
- Optimizer: Adam for Vision Transformer, Adamp for Efficientnet-b4 
- Learning rate scheduler: CosineAnnealingWarmRestarts 
- Time for training: 1.5hr for 1 epoch

6. Postprocessing
- Our team include the MD specialist in Psychiatry. 
- In case of human scorers, the sleep stage is determined by the “context” of sleep in addition to the waveform of the graph.
- We made the post-processing algorithm take this contextual factor into account.

7. References
- https://arxiv.org/abs/2002.12047 
- https://arxiv.org/abs/2010.11929 
- https://developer.nvidia.com/blog/mixed-precision-training-deep-neural-networks/ 
- https://github.com/clovaai/AdamP 
- https://arxiv.org/pdf/1905.11946
