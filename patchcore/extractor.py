"""
This module provides a feature extractor which extract intermediate features
from a neural network model and reshape it to a convenient format for PatchCore.
"""

import warnings

import numpy as np
import rich
import rich.progress
import torch
import thop


# Ignore "UserWarning" because many UserWarnings will be raised
# when calling `thop.profile` function (because of custom layers).
warnings.simplefilter("ignore", UserWarning)


class FeatureExtractor:
    """
    A class to extract intermediate features from a NN model
    and reshape it to a convenient format for PatchCore.

    Example:
    >>> model     = "pytorch/vision:v0.10.0" # Or your custom model
    >>> dataset   = MVTecAD()
    >>> extractor = FeatureExtractor(model)
    >>> 
    >>> extractor.transform(dataset)
    """
    def __init__(self, model, repo, device):
        """
        Constructor of the featureExtractor class.

        Args:
            model  (str/torch.nn.Module): A base NN model.
            repo   (str)                : Repository name which provides the model.
            device (str)                : Device name (e.g. "cuda").
        """
        def load_model(repo, model):
            try   : return torch.hub.load(repo, model, verbose=False, pretrained=True)
            except: return torch.hub.load(repo, model, verbose=False)

        if isinstance(model, str):
            self.model = load_model(repo, model)
            self.name  = model
            macs, pars = thop.profile(self.model, inputs=(torch.randn(1, 3, 224, 224),), verbose=False)
            rich.print("Model summary (assume 3x224x224 input):")
            rich.print("  - [green]repo[/green]: [magenta]{:s}[/magenta]".format(repo))
            rich.print("  - [green]name[/green]: [magenta]{:s}[/magenta]".format(model))
            rich.print("  - [green]pars[/green]: [magenta]{:,}[/magenta]".format(int(pars)))
            rich.print("  - [green]macs[/green]: [magenta]{:,}[/magenta]".format(int(macs)))
        else:
            rich.print("Custom model specified")
            self.model = model
            self.name  = model.__class__.__name__

        # Send model to the device.
        self.device = device
        self.model.to(device)

        # Freeze the model.
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        # Embed hook functions to the mode for extracting intermediate features.
        self.model = self.embed_hooks(self.model)

    def forward(self, x):
        """
        Extract intermediate feature from a single batch data.
        Note that the output tensor is still a tensor of 4-rank (not reshaped).

        Args:
            x (torch.Tensor): Input tensor of shape (N, C_in, H_in, W_in).

        Returns:
            embeddings (torch.Tensor): Embedding representation of shape (N, C_out, H_out, W_out).
        """
        def concat_embeddings(*xs):
            """
            Concatenate the given intermediate features with resizing.

            Args:
                x[i] (torch.Tensor): Input tensor of i-th argument with shape (N, C_i, H_i, W_i).

            Returns:
                z (torch.Tensor): Concatenated tensor with shape (N, sum(C_i), max(H_i), max(W_i)).
            """
            # Compute maximum shape.
            H_max = max([x.shape[2] for x in xs])
            W_max = max([x.shape[3] for x in xs])

            # Create resize function instance.
            resizer = torch.nn.Upsample(size=(H_max, W_max), mode="nearest")

            # Apply resize function for all input tensors.
            zs = [resizer(x) for x in xs]

            # Concatenate in the channel dimention and return it.
            return torch.cat(zs, dim=1)

        # Extract features using hook mechanism.
        self.features = []
        _ = self.model(x.to(self.device))

        # Apply smoothing (3x3 average pooling) to the features.
        smoothing = torch.nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        features  = [smoothing(feature).cpu() for feature in self.features]

        # Concatenate intermediate features.
        embedding = concat_embeddings(*features)

        return embedding

    def transform(self, data, description="Extracting...", **kwargs):
        """
        Extract features from the given data.
        This function can handle 2 types of input:
            (1) dataset (torch.utils.data.Dataset),
            (2) tensor  (torch.Tensor),
        where the shape of the tensor is (N, C, H, W).

        Args:
            data        (Dataset/Tensor): Input data.
            description (str)           : Message shown on the progress bar.
            kwargs      (dict)          : Keyword arguments for DataLoader class constructor.

        Returns:
            embeddings (torch.Tensor): Embedding representations of shape (N*H*W, C).
        """
        def flatten_NHW(tensor):
            """
            Flatten the given tensor of rank-4 with shape (N, C, H, W)
            in terms of N, H and W, and returns a matrix with shape (N*H*W, C).

            Args:
                tensor (torch.Tensor): Tensor of shape (N, C, H, W).

            Returns:
                matrix (torch.Tensor): Tensor of shape (N*H*W, C).
            """
            return tensor.permute((0, 2, 3, 1)).flatten(start_dim=0, end_dim=2)

        # Case 1: input data is a dataset.
        if isinstance(data, torch.utils.data.Dataset):

            # Create data loader.
            dataloader = torch.utils.data.DataLoader(data, **kwargs, pin_memory=True)

            # Extract features for each batch.
            embeddings_list = list()
            for x, _, _, _, _ in rich.progress.track(dataloader, description=description):
                embedding = self.forward(x)
                embeddings_list.append(flatten_NHW(embedding).cpu().numpy())

            # Concat results for each batch and 
            return np.concatenate(embeddings_list, axis=0)

        # Case 2: input data is a single tensor (i.e. single batch).
        elif isinstance(data, torch.Tensor):
            embedding = self.forward(data)
            return flatten_NHW(embedding).cpu().numpy()

    def embed_hooks(self, model):
        """
        Embed hook functions to a NN model for extracting intermediate features.
        """
        # Hook function for capturing intermediate features.
        def hook(module, input, output):
            self.features.append(output)

        RESNET_FAMILIES = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
                           "resnext101_32x8d", "resnext50_32x4d",
                           "wide_resnet50_2", "wide_resnet101_2"]

        DEEPLAB_RESNET = ["deeplabv3_resnet50", "deeplabv3_resnet101"]

        DENSENET_FAMILIES = ["densenet121", "densenet161", "densenet169", "densenet201"]

        VGG_FAMILIES = ["vgg11", "vgg11_bn", "vgg13", "vgg13_bn",
                        "vgg16", "vgg16_bn", "vgg19", "vgg19_bn"]

        if self.name in RESNET_FAMILIES:
            model.layer2[-1].register_forward_hook(hook)
            model.layer3[-1].register_forward_hook(hook)

        elif self.name in DEEPLAB_RESNET:
            model.backbone.layer2[-1].register_forward_hook(hook)
            model.backbone.layer3[-1].register_forward_hook(hook)

        elif self.name in DENSENET_FAMILIES:
            model.features.denseblock2.register_forward_hook(hook)
            model.features.denseblock3.register_forward_hook(hook)

        # In the case of VGG, register the 2nd and 3rd MaxPool counted from the bottom.
        elif self.name in VGG_FAMILIES:
            num_maxpool = 0
            for idx, module in reversed(list(enumerate(model.features))):
                if module.__class__.__name__ == "MaxPool2d":
                    num_maxpool += 1
                    if num_maxpool in [2, 3]:
                        model.features[idx].register_forward_hook(hook)

        # Network proposed by the following paper:
        #   I. Zeki Yalniz, H. Jegou, K. Chen, M. Paluri, and D. Mahajan,
        #   "Billion-scale semi-supervised learning for image classification", CVPR, 2019.
        #   <https://arxiv.org/abs/1905.00546>
        #
        # PyTorch Hub: <https://pytorch.org/hub/facebookresearch_semi-supervised-ImageNet1K-models_resnext>
        # GitHub repo: <https://github.com/facebookresearch/semi-supervised-ImageNet1K-models>
        elif self.name in ["resnet18_swsl", "resnet50_swsl", "resnet18_ssl", "resnet50_ssl"]:
            model.layer2.register_forward_hook(hook)
            model.layer3.register_forward_hook(hook)

        # Raise an error if the given network is unknown.
        else: RuntimeError("unknown neural network: no hooks registered")

        return model
