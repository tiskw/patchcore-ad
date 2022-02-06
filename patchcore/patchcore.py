"""
This module provides a class for PatchCore algorithm.
"""

import argparse
import os
import pathlib

import cv2 as cv
import numpy as np
import rich
import rich.progress
import sklearn.metrics
import scipy.ndimage
import torch
import torchvision

from patchcore.dataset   import MVTecAD
from patchcore.extractor import FeatureExtractor
from patchcore.knnsearch import KNNSearcher
from patchcore.utils     import Timer


class PatchCore:
    """
    PyTorch implementation of the PatchCore anomaly detection [1].

    PatchCore algorithm can be divided to 2 steps, (1) feature extraction from NN model,
    and (2) k-NN search including coreset subsampling. The procedure of the step (1) is
    written in `extractor.FeatureExtractor`, and the step (2) is written in ``.

    Reference:
        [1] K. Roth, L. Pemula, J. Zepeda, B. Scholkopf, T. Brox, and P. Gehler,
            "Towards Total Recall in Industrial Anomaly Detection", arXiv, 2021.
            <https://arxiv.org/abs/2106.08265>
    """
    def __init__(self, model, repo, device, sampling_ratio=0.001):
        """
        Constructor of the PatchCore class.

        Args:
            model  (str/torch.nn.Module): A base NN model.
            repo   (str)                : Repository name which provides the model.
            device (str)                : Device type used for NN inference.
            outdir (str)                : Path to output directory.

        Notes:
            The arguments `model` and `repo` are passed to `torch.hub.load`
            function if the `model` is not a `torch.Module` instance.
        """
        self.device = device

        # Create feature extractor instance.
        self.extractor = FeatureExtractor(model, repo, device)

        # Create k-NN searcher instance.
        self.searcher = KNNSearcher(sampling_ratio=sampling_ratio)

    def fit(self, dataset, batch_size, num_workers=0):
        """
        Args:
            dataset (torchvision.utils.data.Dataset): Dataset.
        """
        # Step (1): feature extraction from the NN model.
        rich.print("\n[yellow][Training 1/2: feature extraction][/yellow]")
        embeddings = self.extractor.transform(dataset, batch_size=batch_size, num_workers=num_workers)
        rich.print("Embeddings dimentions: [magenta]%s[/magenta]" % str(embeddings.shape))

        # Step (2): preparation for k-NN search.
        rich.print("\n[yellow][Training 2/2: preparation for k-NN search][/yellow]")
        self.searcher.fit(embeddings)

    def score(self, dataset, n_neighbors, dirpath_out, num_workers=0):
        """
        Args:
            dataset (torchvision.utils.data.Dataset): Dataset.
            n_neighbors (int): Number of neighbors to be computed in k-NN search.
            dirpath_out (str): Directory path to dump detection results.
            num_workers (int): Number of available CPUs.
        """
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=num_workers, pin_memory=True)

        # Create timer instances.
        timer_feature_ext = Timer()
        timer_k_nn_search = Timer()

        # Initialize results.
        list_true_px_lvl, list_true_im_lvl = (list(), list())
        list_pred_px_lvl, list_pred_im_lvl = (list(), list())

        # Create output firectory if not exists.
        dirpath_images = pathlib.Path(dirpath_out) / "samples"
        dirpath_images.mkdir(parents=True, exist_ok=True)

        rich.print("\n[yellow][Test: anomaly detection inference][/yellow]")
        for x, gt, label, filepath, x_type in rich.progress.track(dataloader, description="Processing..."):

            # Extract embeddings.
            with timer_feature_ext:
                embedding = self.extractor.transform(x)

            # Compute nearest neighbor point and it's score (L2 distance).
            with timer_k_nn_search:
                score_patches, _ = self.searcher.predict(embedding, k=n_neighbors)

            anomaly_map_rw, score = self.compute_anomaly_scores(score_patches, x.shape)

            # Add pixel level scores.
            list_true_px_lvl.extend(gt.cpu().numpy().astype(int).ravel())
            list_pred_px_lvl.extend(anomaly_map_rw.ravel())

            # Add image level scores.
            list_true_im_lvl.append(label.cpu().numpy()[0])
            list_pred_im_lvl.append(score)

            # Save anomaly maps as images.
            self.save_anomaly_map(dirpath_images, anomaly_map_rw, filepath[0], x_type[0])

        rich.print("\n[yellow][Test: score calculation][/yellow]")
        image_auc = sklearn.metrics.roc_auc_score(list_true_im_lvl, list_pred_im_lvl)
        pixel_auc = sklearn.metrics.roc_auc_score(list_true_px_lvl, list_pred_px_lvl)
        rich.print("Total image-level auc-roc score: [magenta]%.6f[/magenta]" % image_auc)
        rich.print("Total pixel-level auc-roc score: [magenta]%.6f[/magenta]" % pixel_auc)

        rich.print("\n[yellow][Test: inference time][/yellow]")
        t1 = timer_feature_ext.mean()
        t2 = timer_k_nn_search.mean()
        rich.print("Feature extraction: [magenta]%.4f sec/image[/magenta]" % t1)
        rich.print("Anomaly map search: [magenta]%.4f sec/image[/magenta]" % t2)
        rich.print("Total infer time  : [magenta]%.4f sec/image[/magenta]" % (t1 + t2))

    def predict(self, x, n_neighbors):
        """
        Returns prediction results.

        Args:
            x           (np.ndarray): Input matrix with shape (n_samples, n_fesatures).
            n_neighbors (int)       : Number of neighbors to be returned.
        """
        # Extract embeddings.
        embedding = self.extractor.transform(x)

        # Compute nearest neighbor point and it's score (L2 distance).
        score_patches, _ = self.searcher.predict(embedding, k=n_neighbors)

        # Compute anomaly map and it's re-weighting.
        anomaly_map_rw, _ = self.compute_anomaly_scores(score_patches, x.shape)

        return anomaly_map_rw

    def load(self, filepath): 
        """
        Load trained model.

        Args:
            filepath (str): path to the trained file.
        """
        self.searcher.load(filepath)

    def save(self, filepath):
        """
        Save trained model.

        Args:
            filepath (str): path to the trained file.
        """
        self.searcher.save(filepath)

    def compute_anomaly_scores(seld, score_patches, x_shape):
        """
        Returns anomaly index from the results of k-NN search.

        Args:
            score_patches (np.ndarary): Results of k-NN search with shape (1, h, w, neighbors).
            x_shape       (tuple)     : Shape of input image with shape (1, c, H, W).
        """
        # Anomaly map is defined as a map of L2 distance from the nearest neibours.
        # NOTE: The magic number (28, 28) should be removed!
        anomaly_score = score_patches.reshape((28, 28, -1))

        # Refine anomaly map.
        anomaly_maps = [anomaly_score[:, :, n] for n in range(anomaly_score.shape[2])]
        anomaly_maps = [cv.resize(amap, (x_shape[3], x_shape[2])) for amap in anomaly_maps]
        anomaly_maps = [scipy.ndimage.gaussian_filter(amap, sigma=4) for amap in anomaly_maps]
        anomaly_maps = np.array(anomaly_maps, dtype=np.float64)
        anomaly_map  = anomaly_maps[0, :, :]

        # Anomaly map re-wighting.
        # We applied log-softmax like processing to the computation of scale factor
        # for avoiding the overflow of floating numbers.
        normalized_exp = np.exp(anomaly_maps - np.max(anomaly_maps, axis=0))
        anomaly_map_rw = anomaly_map * (1.0 - normalized_exp[0, :, :] / np.sum(normalized_exp, axis=0))

        # Compute image level score.
        i, j  = np.unravel_index(np.argmax(anomaly_map), anomaly_map.shape)
        score = anomaly_map_rw[i, j]

        return (anomaly_map_rw, score)

    def save_anomaly_map(self, dirpath, anomaly_map, filepath, x_type):
        """
        Args:
            anomaly_map   (np.ndarray): Anomaly detection result with the same
                                        size as the input image.
            filepath      (str)       : Path of the input image.
            x_type        (str)       : Anomaly type (e.g. "good", "crack", etc).
        """
        def min_max_norm(image):
            a_min, a_max = image.min(), image.max()
            return (image - a_min) / (a_max - a_min)    

        def cvt2heatmap(gray):
            return cv.applyColorMap(np.uint8(gray), cv.COLORMAP_JET)

        # Get output directory.
        dirpath = pathlib.Path(dirpath)
        dirpath.mkdir(parents=True, exist_ok=True)

        # Get file name.
        filename = os.path.basename(filepath)

        # Normalize anomaly map for easier visualization.
        anomaly_map_norm = cvt2heatmap(255 * min_max_norm(anomaly_map))
        original_image   = cv.resize(cv.imread(filepath), anomaly_map_norm.shape[:2])
        output_image     = (anomaly_map_norm / 2 + original_image / 2).astype(np.uint8)

        # Save the normalized anomaly map as image.
        cv.imwrite(str(dirpath / f"{x_type}_{filename}.jpg"), output_image)

        # Save raw anomaly score as npy file.
        np.save(str(dirpath / f"{x_type}_{filename}.npy"), anomaly_map)
