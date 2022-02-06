Experiments
================================================================================

This directory contains experiment results and their summaries.

Experiment 1: Comparison with the original paper
--------------------------------------------------------------------------------

- **Purpose**: Our purpose is to check that our implementation correctly
  reproduced the PatchCore algorithm.

- **What we've done**: Evaluate our implementation on the MVTec AD dataset
  and compare it with the results written in the original paper.

- **Conclusion**: The score of our implementation is quite close to
  the paper's score. Therefore, our implementation may not have a serious issue.

See [this document](summary_comparison_with_the_paper.md) for details.

<div align="center">
    <img width="40%" src="figures/MVTecAD_averaged_image-level_roc_auc_score.svg" />
    <img width="40%" src="figures/MVTecAD_averaged_pixel-level_roc_auc_score.svg" />
</div>


Experiment 2: Comparison of backborn networks
--------------------------------------------------------------------------------

- **Purpose**: It's quite easy to swap the backbone network in the PatchCore
  algorithm (default: Wide ResNet50 x2). It's meaningful to find a good
  backbone network that shows a good performance-speed tradeoff from
  an application viewpoint.

- **What we've done**: Try several backbone networks and evaluate their
  average image/pixel-level scores in the MVTecAD dataset.

- **Conclusion**: The smaller ResNet (ResNet18, ResNet34) shows enough good
  scores even for their small computational cost. On the other hand, very
  deep ResNet (ResNet101, ResNet152) shows lower performance than ResNet50.
  A current tentative hypothesis is that the features used in the PatchCore
  algorithm are too deep (too far from input) and don't have enough
  high resolution (raw) features in them. In other words, we should
  add shallower features in the case of very deep neural networks
  like ResNet101/ResNet152 for exceeding ResNet50's score.

<div align="center">
    <img width="40%" src="figures/MVTecAD_image-level_roc_auc_score_backbones.svg" />
    <img width="40%" src="figures/MVTecAD_pixel-level_roc_auc_score_backbones.svg" />
</div>

See [this document](summary_comparison_with_backbones.md) for details.


Experiment 3: Comparison of pre-trainings
--------------------------------------------------------------------------------

- **Purpose**: We hyposesize that well-trained neural network achieves
  higher performance on PatchCore anomaly detection.

- **What we've done**: We used several networks that are pre-training in
  a different dataset or different method as a backbone of PatchCore,
  and evaluate their performance on the MVTec AD dataset.

- **Conclusion**: We found quite interesting observations, however,
  we cannot conclude it at this moment. We would say that the normal ImageNet
  pre-trained model seems to be enough good for PatchCore purpose.

<div align="center">
    <img width="60%" src="figures/MVTecAD_ResNet50_with_different_pretrainings.svg" />
</div>
