"""
This module provides a PyTorch implementation of the MVTec AD dataset.
"""

# Import standard libraries.
import pathlib

# Import third-party packages.
import numpy as np
import PIL.Image
import torch
import torchvision


class MVTecAD(torch.utils.data.Dataset):
    """
    PyTorch implementation of the MVTec AD dataset [1].

    Reference:
        [1] <https://www.mvtec.com/company/research/datasets/mvtec-ad>
    """
    def __init__(self, root="data", split="train", transform_im=None, transform_gt=None,
                 load_shape=(256, 256), input_shape=(224, 224),
                 im_mean=(0.485, 0.456, 0.406), im_std=(0.229, 0.224, 0.225)):
        """
        Constructor of MVTec AD dataset.

        Args:
            root         (str)  : Dataset directory.
            split        (str)  : The dataset split, supports `train`, or `test`.
            transform_im (func) : Transform for the input image.
            transform_gt (func) : Transform for the ground truth image.
            load_shape   (tuple): Shape of the loaded image.
            input_shape  (tuple): Shape of the input image.
            im_mean      (tuple): Mean of image (3 channels) for image normalization.
            im_std       (tuple): Standard deviation of image (3 channels) for image normalization.

        Notes:
            The arguments `load_shape`, `input_shape`, `im_mean`, and `im_std` are used
            only if the `transform_im` or `transform_gt` is None.
        """
        self.root = pathlib.Path(root)

        # Set directory path of input and ground truth images.
        if split == "train":
            self.dir_im = self.root / "train"
            self.dir_gt = None
        elif split == "test":
            self.dir_im = self.root / "test"
            self.dir_gt = self.root / "ground_truth"

        # The value of `split` should be either of "train" or "test".
        else: raise ValueError("Error: argument `split` should be 'train' or 'test'.")

        # Use default transform if no transform specified.
        args = (load_shape, input_shape, im_mean, im_std)
        self.transform_im = self.default_transform_im(*args) if transform_im is None else transform_im
        self.transform_gt = self.default_transform_gt(*args) if transform_gt is None else transform_gt

        self.paths_im, self.paths_gt, self.labels, self.anames = self.load_dataset(self.dir_im, self.dir_gt)

    def __getitem__(self, idx):
        """
        Returns idx-th data.

        Args:
            idx (int): Index of image to be returned.
        """
        path_im = self.paths_im[idx]  # Image file path.
        path_gt = self.paths_gt[idx]  # Ground truth file path.
        flag_an = self.labels[idx]    # Anomaly flag (good : 0, anomaly : 1).
        name_an = self.anames[idx]    # Anomaly name.

        # Load input image.
        img = PIL.Image.open(str(path_im)).convert("RGB")

        # If good data, use zeros as a ground truth image.
        if flag_an == 0:
            igt = PIL.Image.fromarray(np.zeros(img.size[::-1], dtype=np.uint8))

        # Otherwise, load ground truth data.
        elif flag_an == 1:
            igt = PIL.Image.open(str(path_gt)).convert("L")

        # Anomaly flag should be either of 0 or 1.
        else: raise ValueError("Error: value of `flag_an` should be 0 or 1.")

        # Size of the input and ground truth image should be the same.
        assert img.size == igt.size, "image.size != igt.size !!!"

        # Apply transforms.
        img = self.transform_im(img)
        igt = self.transform_gt(igt)

        return (img, igt, flag_an, str(path_im), name_an)

    def __len__(self):
        """
        Returns number of data.
        """
        return len(self.paths_im)

    def load_dataset(self, dir_im, dir_gt):
        """
        Load dataset.

        Args:
            dir_im (pathlib.Path): Path to the input image directory.
            dir_gt (pathlib.Path): Path to the ground truth image directory.
        """
        paths_im = list()  # List of image file paths.
        paths_gt = list()  # List of ground truth file paths.
        flags_an = list()  # List of anomaly flags (good : 0, anomaly : 1).
        names_an = list()  # List of anomaly names.

        for subdir in sorted(dir_im.iterdir()):

            # Name of the sub directory is the same as the anomaly name.
            defect_name = subdir.name

            # Case 1: good data which have only input image.
            if defect_name == "good":

                # Get input image paths (good data doesn't have ground truth image).
                paths = sorted((dir_im / defect_name).glob("*.png"))

                # Update attributes.
                paths_im += paths
                paths_gt += len(paths) * [None]
                flags_an += len(paths) * [0]
                names_an += len(paths) * [defect_name]

            # Case 2: not good data which have both input and ground truth images.
            else:

                # Get input and ground truth image paths.
                paths1 = sorted((dir_im / defect_name).glob("*.png"))
                paths2 = sorted((dir_gt / defect_name).glob("*.png"))

                # Update attributes.
                paths_im += paths1
                paths_gt += paths2
                flags_an += len(paths1) * [1]
                names_an += len(paths2) * [defect_name]

        # Number of input image and ground truth image sould be the same.
        assert len(paths_im) == len(paths_gt), "Something wrong with test and ground truth pair!"

        return (paths_im, paths_gt, flags_an, names_an)

    def default_transform_im(self, load_shape, input_shape, im_mean, im_std):
        """
        Returns default transform for the input image of MVTec AD dataset.
        """
        return torchvision.transforms.Compose([
            torchvision.transforms.Resize(load_shape),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.CenterCrop(input_shape),
            torchvision.transforms.Normalize(mean=im_mean, std=im_std),
        ])

    def default_transform_gt(self, load_shape, input_shape, im_mean, im_std):
        """
        Returns default transform for the ground truth image of MVTec AD dataset.
        """
        return torchvision.transforms.Compose([
            torchvision.transforms.Resize(load_shape),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.CenterCrop(input_shape),
        ])


class MVTecADImageOnly(torch.utils.data.Dataset):
    """
    Dataset class that is quite close to MVTecAD, but works
    if no ground truth image is available. This class will be
    used for user defined datasets.
    """
    def __init__(self, root="data", transform=None,
                 load_shape=(256, 256), input_shape=(224, 224),
                 im_mean=(0.485, 0.456, 0.406), im_std=(0.229, 0.224, 0.225)):
        """
        Constructor of MVTec AD dataset.

        Args:
            root        (str)  : Image directory.
            transform   (func) : Transform for the input image.
            load_shape  (tuple): Shape of the loaded image.
            input_shape (tuple): Shape of the input image.
            im_mean     (tuple): Mean of image (3 channels) for image normalization.
            im_std      (tuple): Standard deviation of image (3 channels) for image normalization.

        Notes:
            The arguments `load_shape`, `input_shape`, `im_mean`, and `im_std` are used
            only if the `transform_im` or `transform_gt` is None.
        """
        self.root = pathlib.Path(root)

        # Use default transform if no transform specified.
        args = (load_shape, input_shape, im_mean, im_std)
        self.transform = self.default_transform(*args) if transform is None else transform

        self.paths = [path for path in self.root.glob("**/*") if path.suffix in [".jpg", ".png"]]

    def __getitem__(self, idx):
        """
        Returns idx-th data.

        Args:
            idx (int): Index of image to be returned.
        """

        # Load input image.
        path = self.paths[idx]
        img  = PIL.Image.open(str(path)).convert("RGB")

        # Apply transforms.
        img = self.transform(img)

        # Returns only image while keeping the same interface as the MVTecAD class.
        return (img, 0, 0, str(path), 0)

    def __len__(self):
        """
        Returns number of data.
        """
        return len(self.paths)

    def default_transform(self, load_shape, input_shape, im_mean, im_std):
        """
        Returns default transform for the input image of MVTec AD dataset.
        """
        return torchvision.transforms.Compose([
            torchvision.transforms.Resize(load_shape),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.CenterCrop(input_shape),
            torchvision.transforms.Normalize(mean=im_mean, std=im_std),
        ])


if __name__ == "__main__":

    # Test the training data.
    dataset = MVTecAD("data/mvtec_anomaly_detection/hazelnut", "train")
    print(dataset[0])

    # Test the test data.
    dataset = MVTecAD("data/mvtec_anomaly_detection/hazelnut", "test")
    print(dataset[0])
