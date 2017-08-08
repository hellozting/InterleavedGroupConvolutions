# Interleaved Group Convolutions

This project contains the code implementation used for the experiments in the paper:

>  Interleaved Group Convolutions. Ting Zhang, Guo-Jun Qi, Bin Xiao, and Jingdong Wang. In International Conference on Computer Vision (ICCV), 2017.
arXiv preprint [arXIV:1707.02725](https://arxiv.org/pdf/1707.02725.pdf) (2017)

If you use this code for research purposes, please cite the aforementioned paper in any resulting publication.

## Introduction
In this work, we present a simple and modularized neural network architecture, named interleaved group convolutional neural networks (IGCNets). The main point lies in a novel building block, a pair of two successive interleaved group convolutions: primary group convolution and secondary group convolution. The two group convolutions are complementary.

![IGC](visualize/paper/igc.png)
>  Illustrating the interleaved group convolution, with L = 2 primary partitions and M = 3 secondary partitions. The convolution for each primary partition in primary group convolution is spatial. The convolution for each secondary partition in secondary group convolution is point-wise (1 × 1).

Our motivation comes from the four branch presentation of regular convolution illustrated in the following picture.

![RC](visualize/paper/regularconvmultibranch.png)
> (a) Regular convolution. (b) Four-branch representation of the regular convolution. The shaded part in (b), we call crosssummation, is equivalent to a three-step transformation: permutation, secondary group convolution, and permutation back.

## Results

![ImageNet](visualize/paper/ImagenetResults.png)
>  Imagenet classiﬁcation results of a ResNet of depth 18 and our approach. The network structure for ResNet can be found in [10]. Both ResNets and our networks contain four stages, and when down-sampling is performed, the channel number is doubled. For ResNets, C is the channel number at the ﬁrst stage. For our networks except IGC-L100M2+Ident., we double the channel number by doubling M and keeping L unchanged. For IGCL100M2+Ident., we double the channel number by doubling L and keeping M unchanged. 

## Requirements

## How to Train

## Reference

## Citation
