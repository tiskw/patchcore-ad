"""
PatchCore anomaly detection: main script for user custom dataset
"""

# Import standard libraries.
import argparse

# Import third-party packages.
import numpy as np
import rich
import rich.progress
import torch

# Import custom modules.
from patchcore.dataset   import MVTecADImageOnly
from patchcore.patchcore import PatchCore
from patchcore.utils     import auto_threshold


def parse_args():
    """
    Parse command line arguments.
    """
    fmtcls = lambda prog: argparse.HelpFormatter(prog,max_help_position=50)
    parser = argparse.ArgumentParser(description=__doc__, add_help=False, formatter_class=fmtcls)

    # Required arguments.
    parser.add_argument("mode", choices=["train", "predict", "thresh"], help="running mode")

    # Optional arguments for dataset configuration.
    group1 = parser.add_argument_group("dataset options")
    group1.add_argument("-i", "--input", metavar="PATH", default="data/mvtec_ad/bottle/train",
                        help="input file/directory path")
    group1.add_argument("-o", "--output", metavar="PATH", default="output",
                        help="output file/directory path")
    group1.add_argument("-t", "--trained", metavar="PATH", default="index.faiss",
                        help="training results")
    group1.add_argument("-b", "--batch_size", metavar="INT", default=16,
                        help="training batch size")
    group1.add_argument("-l", "--load_size", metavar="INT", default=224,
                        help="size of loaded images")
    group1.add_argument("-n", "--input_size", metavar="INT", default=224,
                        help="size of images passed to NN model")

    # Optional arguments for neural network configuration.
    group2 = parser.add_argument_group("network options")
    group2.add_argument("-m", "--model", metavar="STR", default="wide_resnet50_2",
                        help="name of a neural network model")
    group2.add_argument("-r", "--repo", metavar="STR", default="pytorch/vision:v0.11.3",
                        help="repository of the neural network model")

    # Optional arguments for anomaly detection algorithm.
    group3 = parser.add_argument_group("algorithm options")
    group3.add_argument("-k", "--n_neighbors", metavar="INT", type=int, default=3,
                        help="number of neighbors to be searched")
    group3.add_argument("-s", "--sampling_ratio", metavar="FLT", default=0.01,
                        help="ratio of coreset sub-sampling")

    # Optional arguments for thresholding.
    group4 = parser.add_argument_group("thresholding options")
    group4.add_argument("-e", "--coef_sigma", metavar="FLT", type=float, default=5.0,
                        help="coefficient of sigma when computing threshold (= mean + coef * sigma)")

    # Optional arguments for visualization.
    group5 = parser.add_argument_group("visualization options")
    group5.add_argument("-c", "--contour", metavar="FLT", type=float, default=None,
                        help="visualize contour map instead of heatmap using the given threshold")

    # Other optional arguments.
    group6 = parser.add_argument_group("other options")
    group6.add_argument("-d", "--device", metavar="STR", default="auto",
                        help="device name (e.g. 'cuda')")
    group6.add_argument("-w", "--num_workers", metavar="INT", type=int, default=1,
                        help="number of available CPUs")
    group6.add_argument("-h", "--help", action="help",
                        help="show this help message and exit")

    return parser.parse_args()


def main(args):
    """
    Main function for running training/test procedure.

    Args:
        args (argparse.Namespace): Parsed command line arguments.
    """
    rich.print(r"[yellow][Command line arguments][/yellow]")
    rich.print(vars(args))

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create PatchCore model instance.
    model = PatchCore(args.model, args.repo, args.device, args.sampling_ratio)

    # Arguments required for dataset creation.
    # These arguments are mainly used for the transformations applied to
    # the input images and ground truth images. Details of the transformations
    # are written in the MVTecAD dataset class (see patchcore/dataset.py).
    dataset_args = {
        "load_shape" : (args.load_size, args.load_size),
        "input_shape": (args.input_size, args.input_size),
        "im_mean"    : (0.485, 0.456, 0.406),
        "im_std"     : (0.229, 0.224, 0.225),
        # The above mean and standard deviation is a values of the ImageNet dataset.
        # These values are required because the NN models pre-trained with ImageNet
        # assume that the input image is normalized in terms of ImageNet dataset.
    }

    if args.mode == "train":

        # Prepare dataset.
        dataset = MVTecADImageOnly(args.input, **dataset_args)

        # Train model.
        model.fit(dataset, args.batch_size, args.num_workers)

        # Save training result.
        model.save(args.trained)

    elif args.mode == "predict":

        # Load trained model.
        model.load(args.trained)

        # Prepare dataset.
        dataset = MVTecADImageOnly(args.input, **dataset_args)
        dloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=args.num_workers, pin_memory=True)

        for x, gt, label, filepath, x_type in rich.progress.track(dloader, description="Processing..."):

            # Run prediction and get anomaly heatmap.
            anomaly_map_rw = model.predict(x, args.n_neighbors)

            # Save anomaly heatmap (JPG image and NPY file).
            model.save_anomaly_map(args.output, anomaly_map_rw, filepath[0], x_type[0], contour=args.contour)

    elif args.mode == "thresh":

        # Load trained model.
        model.load(args.trained)

        # Prepare dataset.
        dataset = MVTecADImageOnly(args.input, **dataset_args)
        dloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=args.num_workers, pin_memory=True)

        # Initialize the anomaly scores.
        scores = list()

        # Compute max value of the anomaly heatmaps.
        for x, gt, label, filepath, x_type in rich.progress.track(dloader, description="Processing..."):

            # Run prediction and get anomaly heatmap.
            anomaly_map_rw = model.predict(x, args.n_neighbors)

            # Append the anomaly score.
            scores.append(np.max(anomaly_map_rw))

        # Compute threshold.
        thresh, score_mean, score_std = auto_threshold(scores, args.coef_sigma)

        print("Anomaly threshold = %f" % thresh)
        print("  - score_mean = %f" % score_mean)
        print("  - score_std  = %f" % score_std)


if __name__ == "__main__":
    main(parse_args())
