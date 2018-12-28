# Semantic_Segmentation
Semantic Segmentation on the dataset of surgical segmentation used by the MICCAI competitions

### What is Semantic Segmentation?

Semantic segmentation is a natural step in the progression from coarse to fine inference:
1. The origin could be located at classification, which consists of making a prediction for a whole input.
2. The next step is localization / detection, which provide not only the classes but also additional information regarding the spatial location of those classes.
3. Finally, semantic segmentation achieves fine-grained inference by making dense predictions inferring labels for every pixel, so that each pixel is labeled with the class of its enclosing object ore region.

### Dataset Used
The dataset used to train this model is the same used in the following repository:

https://github.com/ternaus/TernausNet


### Model Used Here

Semantic segmentation model used here is a custom one built with the following encoder and decoder layers:
![Architecture of Neural Network for Segmentation](https://raw.githubusercontent.com/yashdevikar/Semantic_Segmentation/master/net.png)
