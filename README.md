## Knowledge Distillation in Semantic Segmentation 
 This is Implementation of [An Efficient Knowledge Distillation Architecture for Real-time Semantic Segmentation](https://drive.google.com/file/d/1wrWg54G1ex-8WRYVMGziWTapXFsFMEW0/view?usp=drivesdk).

### Method Diagram
The architecture of both the teacher and student networks is Deeplab-V3 +, although their encoders are different. Teacher network is fixed the during the training process; only the student network will be trained with two distillation losses and cross-entropy loss. The pixel-wise distillation module uses the preReLU feature map of the last convolution layer of the decoder before probability scores to transfer detailed spatial information. The pair-wise distillation module uses the feature map of the last layer of the encoder to create a pair-wise similarity matrix and transfer global information.

<img src="https://github.com/AmirMansurian/KD/blob/main/Images/KD.png"  width="700" height="500" />

### Experimental Results
Results of each distillation method on the [PascalVoc 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) validation set with two different backbones. Results are average of 3 runs with different random seeds.

<img src="https://github.com/AmirMansurian/KD/blob/main/Images/results.png"   width="700" height="300"/>


### Visualization
Comparison of segmentation results between ground-truth, teacher
prediction, student prediction and prediction after distillation.
<img src="https://github.com/AmirMansurian/KD/blob/main/Images/experiments.png"   width="700" height="600"/>

### How to run
  ```shell
  python train_kd.py --backbone resner18 --dataset pascal  --pa_lambda 1 --pi_lambda 100 
  ```

### Teacher model
Download following pre-trained teacher network and put it into ```./Segmentation/pretrained``` directory
- [ResNet101-DeepLabV3+](https://drive.google.com/open?id=1Pz2OT5KoSNvU5rc3w5d2R8_0OBkKSkLR)

 measure performance on **test** set with [Pascal VOC evaluation server](http://host.robots.ox.ac.uk/pascal/VOC/).
 
 ## Citation
If you use this repository for your research or wish to refer to our distillation method, please use the following BibTeX entry:
```
@article{
}
```

### Acknowledgement
This codebase is heavily borrowed from [A Comprehensive Overhaul of Feature Distillation ](https://github.com/clovaai/overhaul-distillation) and [structure_knowledge_distillation](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwiM3vjGyuD7AhXohf0HHcA4B44QFnoECDIQAQ&url=https%3A%2F%2Fgithub.com%2FirfanICMLL%2Fstructure_knowledge_distillation&usg=AOvVaw2rg7lYss4wqcvKKDH0UWoN). Thanks for their excellent works.
