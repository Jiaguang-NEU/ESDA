
# ESDA

> **Abstract:** 

<p align="middle">
  <img src="figure/main.png">
</p>

### Dependencies

- RTX 3090
- Python 3.8
- PyTorch 1.12.0
- cuda 11.6
- torchvision 0.13.0
- tensorboardX 2.2


### Datasets

- PASCAL-5<sup>i</sup>:  [VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) + [SBD](http://home.bharathh.info/pubs/codes/SBD/download.html)

- COCO-20<sup>i</sup>:  [COCO2014](https://cocodataset.org/#download)
- Put the datasets into the `data/` directory.
- Download the data lists from [BAM](https://github.com/chunbolang/BAM).

### Models

- Download the pre-trained [CLIP(ViT-B/16)](https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt) and put them into the `model/backbone/pretrain_model` directory.


### Scripts

- Change configuration and add weight path to `.yaml` files in `config` directory, then run the `train.sh` file for training or `test.sh` file for testing.

### Performance

Performance comparison with the state-of-the-art approachs in terms of **average** **mIoU** across all folds. 


   | Backbone  | Method       | PASCAL-5i               | COCO-20i                 |
   | --------  | ------------ | ------------------------| ------------------------ |
   | ViT-B     | ClsCLIP      | 56.4                    | 37.0                     |
   | ViT-B     | ESDA(ours)   | 58.9                    | 37.1                     |


   




## References

This repo is mainly built based on [CLIP](https://github.com/openai/CLIP), and [BAM](https://github.com/chunbolang/BAM). Thanks for their great work!

````
This paper is currently being reviewed by the Knowledge Based Systems journal.
