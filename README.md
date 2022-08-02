## Knowledge Distillation in Semantic Segmentation - Pascal VOC
### Deeplab architecture

![Alt Text](https://raw.githubusercontent.com/AmirMansurian/KD/main/Deeplab%20Architecture.png)

### Teacher and Student

|   Model  |  Backbone  | Model size | Epoches | mIOU |
|:----------:|:---------:|:------------:|:------------:|:----------:|
| Deeplab | ResNet 18 |    16.6    |    120    |  67/50 %   |
| Deeplab | ResNet 101 |     59.3M      |   120   |  74/08 %   |


### Settings

|   Teacher  |  Student  | Loss | mIOU |
|:----------:|:---------:|:------------:|:------------:|
| ResNet 101 | ResNet 18 |   CE + Logits    |    68/67 %    |  
| ResNet 101 | ResNet 18 |    CE + feature_maps    |     69/15 %   |
| ResNet 101 | ResNet 18 |    CE + feature_maps + Logits    |    68/84 %   |


### Teacher model
Download following pre-trained teacher network and put it into ```./Segmentation/pretrained``` directory
- [ResNet101-DeepLabV3+](https://drive.google.com/open?id=1Pz2OT5KoSNvU5rc3w5d2R8_0OBkKSkLR)

 measure performance on **test** set with [Pascal VOC evaluation server](http://host.robots.ox.ac.uk/pascal/VOC/).
