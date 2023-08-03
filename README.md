## Adaptive Inter-Class Similarity Distillation for Semantic Segmentation 
 This repository contains the source code of AICSD [(Adaptive Inter-Class Similarity Distillation for Semantic Segmentation )](https://drive.google.com/file/d/1wrWg54G1ex-8WRYVMGziWTapXFsFMEW0/view?usp=drivesdk).

<p align="center">
 <img src="https://raw.githubusercontent.com/AmirMansurian/AICSD/main/Images/pull_figure_main.png"  width="500" height="500"/>
</p>

 Intra-class distributions for each class. Distributions are created by applying softmax to spatial dimension of output prediction of last layer. Similarities between each pair of intra-class distributions have good potential for distillation. Distributions are created from the PASCAL VOC 2012 dataset with 21 category classes.

### Method Diagram
<img src="https://raw.githubusercontent.com/AmirMansurian/AICSD/main/Images/Method_diagram.png"  width="700" height="300" />

**Overall diagram of the proposed AICSD**. Network outputs are flattened into 1D vectors, followed by application of a softmax function to create intra-class distributions. KL divergence is then calculated between each distribution to create inter-class similarity matrices. An MSE loss function is then defined between the ICS matrices of the teacher and student. Also, KL divergence is calculated between the logits of the teacher and student for pixel-wise distillation. To mitigate the negative effects of teacher network, an adaptive weighting loss strategy is used to scale two distillation losses and ross-entropy loss of semantic segmentation. During training, hyperparameter $\alpha$ undergoes adaptive changes and progressively increases with epoch number.

### Performance on PascalVOC2012
Results of each distillation method on the [PascalVoc 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) validation set with two different backbones. Results are average of 3 runs with different random seeds.

<img src="https://github.com/AmirMansurian/KD/blob/main/Images/results.png"   width="700" height="300"/>


### Performance on CityScapes
Results of each distillation method on the [PascalVoc 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) validation set with two different backbones. Results are average of 3 runs with different random seeds.

<img src="https://github.com/AmirMansurian/KD/blob/main/Images/results.png"   width="700" height="300"/>

### Visualization
<img src="https://raw.githubusercontent.com/AmirMansurian/AICSD/main/Images/visualization_2.png"   width="700" height="400"/>

### How to run
  ```shell
  python train_kd.py --backbone resnet18 --dataset pascal  --pa_lambda 1 --pi_lambda 100 
  ```

### Teacher model
Download following pre-trained teacher network and put it into ```pretrained/``` directory
- [ResNet101-DeepLabV3+](https://drive.google.com/open?id=1Pz2OT5KoSNvU5rc3w5d2R8_0OBkKSkLR)

 measure performance on **test** set with [Pascal VOC evaluation server](http://host.robots.ox.ac.uk/pascal/VOC/).
 
 ## Citation
If you use this repository for your research or wish to refer to our distillation method, please use the following BibTeX entry:
```
@article{
}
```

### Acknowledgement
This codebase is heavily borrowed from [A Comprehensive Overhaul of Feature Distillation ](https://github.com/clovaai/overhaul-distillation). Thanks for their excellent work.
