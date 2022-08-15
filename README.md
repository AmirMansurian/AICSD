## Knowledge Distillation in Semantic Segmentation - Pascal VOC
### Deeplab architecture

![Alt Text](https://raw.githubusercontent.com/AmirMansurian/KD/main/Images/Deeplab%20Architecture.png)


### Teacher and Student

|   Model  |  Backbone  | Model size | Epoches | mIOU |
|:----------:|:---------:|:------------:|:------------:|:----------:|
| Deeplab | ResNet 18 |    16.6    |    120    |  67/50 %   |
| Deeplab | ResNet 101 |     59.3M      |   120   |  74/08 %   |


### Settings

|   Teacher  |  Student  | Loss | mIOU |
|:----------:|:---------:|:------------:|:------------:|
| ResNet 101 | ResNet 18 |   CE + all feature_maps    |    67/65 %    | 
| ResNet 101 | ResNet 18 |   CE + backbone feature_maps    |    67/75 %    | 
| ResNet 101 | ResNet 18 |   CE + grad_based    |    67/79 %    |  
| ResNet 101 | ResNet 18 |   CE + Logits    |    68/67 %    |  
| ResNet 101 | ResNet 18 |    CE + ASPP feature_maps    |     69/15 %   |
| ResNet 101 | ResNet 18 |    CE + ASPP + Logits    |    69/38 %   |
| ResNet 101 | ResNet 18 |    CE + last layer feature_maps    |    69/82 %   |


### Visuialization

![Alt Text](https://raw.githubusercontent.com/AmirMansurian/KD/main/Images/input.jpeg)
![Alt Text](https://raw.githubusercontent.com/AmirMansurian/KD/main/Images/feature.jpeg)
![Alt Text](https://raw.githubusercontent.com/AmirMansurian/KD/main/Images/grad.jpeg)

### Teacher model
Download following pre-trained teacher network and put it into ```./Segmentation/pretrained``` directory
- [ResNet101-DeepLabV3+](https://drive.google.com/open?id=1Pz2OT5KoSNvU5rc3w5d2R8_0OBkKSkLR)

 measure performance on **test** set with [Pascal VOC evaluation server](http://host.robots.ox.ac.uk/pascal/VOC/).
