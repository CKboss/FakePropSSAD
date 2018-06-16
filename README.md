## Fake Prop SSAD

This is just a fake SSAD implement by pytorch. It can't run directly. This is only the key part of the network. May different from the real SSAD. 

## How to use

1. Prepare Dataset's feature

Frame level feature were extruct from dataset. Pretrain a TSN or other video classfier you like on the dataset to get every frames feature.
You can use linear interpolation to make every video have same num of features or use slide window.

2. What the Dataset look like

```python

gF,gL,gB,gI = dataset_train.nextbatch(config.batch_size)

```
Where: 

* gF is video's feature
* gL is the video's propoposal's label
* gB is the ground truth of the video action detection
* gI is the index of every proposals ( One video may have more than one proposals, gI[i]~gI[i+1] is the range where label and ground truth from the i th video )


3. Train the network

## Other

All code where inspired by:

```
@article{DBLP:journals/corr/abs-1710-06236,
  author    = {Tianwei Lin and
               Xu Zhao and
               Zheng Shou},
  title     = {Single Shot Temporal Action Detection},
  journal   = {CoRR},
  volume    = {abs/1710.06236},
  year      = {2017},
  url       = {http://arxiv.org/abs/1710.06236},
  archivePrefix = {arXiv},
  eprint    = {1710.06236},
  timestamp = {Wed, 01 Nov 2017 19:05:42 +0100},
  biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1710-06236},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
