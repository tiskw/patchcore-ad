Comparison of pre-trainings
================================================================================

This directory contains experiment results of ResNet50 that are pre-trained
with difference dataset or different method.


Purpose
--------------------------------------------------------------------------------

We hyposesize that well-trained neural network achieves higher performance.


What we've done
--------------------------------------------------------------------------------

We tried ResNet50 with ImageNet pre-training, DeepLab V3 pretrained with COCO,
and SSL/SWSL model pretrained on ImageNet [3]. 
We used these networks as a backbone of PatchCore and evaluate their
performance on the MVTec AD dataset.


Conclution
--------------------------------------------------------------------------------

We found quite interesting observations:

- Simple ImageNet pre-trained ResNet50 (ResNet50 in the following figure)
  shows lower performance in the image-level ROC AUC score, on the other hand,
  it shows higher performance in the pixel-level ROC AUC score.
- One possible reasoning is that the simple ImageNet pre-trained model
  keeps high-resolution (= raw) information in the features than the other
  models which trained on more complex dataset or trained by more complex
  method. High-resolution information may bring higer pixel-lebel score
  (and lower image-level score), therefore it can be an explanation
  of this phenomenon.

We cannot conclude it at this moment and we need more experiments, however,
I would say that the normal ImageNet pre-trained model seems to be enough
good for PatchCore purpose.

<div align="center">
    <img width="60%" src="figures/MVTecAD_ResNet50_with_different_pretrainings.svg" />
</div>
