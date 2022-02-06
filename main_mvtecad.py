"""
PatchCore anomaly detection: main script for MVTec AD dataset
"""

import argparse
import os
import pathlib
import subprocess

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
from patchcore.patchcore import PatchCore
from patchcore.utils     import Timer


def parse_args():
    """
    Parse command line arguments.
    """
    fmtcls = lambda prog: argparse.HelpFormatter(prog,max_help_position=50)
    parser = argparse.ArgumentParser(description=__doc__, add_help=False, formatter_class=fmtcls)

    # Required arguments.
    parser.add_argument("mode", choices=["train", "test", "runall", "summ"],
                        help="running mode")

    # Common optional arguments.
    group0 = parser.add_argument_group("common options")
    group0.add_argument("--datadir", default="data/mvtec_ad",
                        help="path to MVTec AD dataset")

    # Optional arguments for training and test configuration.
    group1 = parser.add_argument_group("train/test options")
    group1.add_argument("--category", default="hazelnut",
                        help="data category (e.g. 'hazelnut')")
    group1.add_argument("--device", metavar="STR", default="auto",
                        help="device name (e.g. 'cuda')")
    group1.add_argument("--model", metavar="STR", default="wide_resnet50_2",
                        help="name of a neural network model")
    group1.add_argument("--repo", metavar="STR", default="pytorch/vision:v0.11.3",
                        help="repository of the neural network model")
    group1.add_argument("--n_neighbors", metavar="INT", type=int, default=3,
                        help="number of neighbors to be searched")
    group1.add_argument("--sampling_ratio", metavar="FLT", default=0.01,
                        help="ratio of coreset sub-sampling")
    group1.add_argument("--outdir", metavar="PATH", default="output",
                        help="output file/directory path")
    group1.add_argument("--batch_size", metavar="INT", default=16,
                        help="training batch size")
    group1.add_argument("--load_size", metavar="INT", default=256,
                        help="size of loaded images")
    group1.add_argument("--input_size", metavar="INT", default=224,
                        help="size of images passed to NN model")
    group1.add_argument("--num_workers", metavar="INT", type=int, default=1,
                        help="number of available CPUs")

    # Optional arguments for running experiments configuration.
    group2 = parser.add_argument_group("runall options")
    group2.add_argument("--dryrun", action="store_true",
                        help="only dump the commands and do nothing")
    group2.add_argument("--test_only", action="store_true",
                        help="run only test procedure")
    group2.add_argument("--no_redirect", action="store_true",
                        help="do not redirect dump messages")

    # Other optional arguments.
    group3 = parser.add_argument_group("other options")
    group3.add_argument("-h", "--help", action="help",
                        help="show this help message and exit")

    return parser.parse_args()


def main_traintest(args):
    """
    Main function for running training/test procedure.

    Args:
        args (argparse.Namespace): Parsed command line arguments.
    """
    rich.print(r"[yellow][Command line arguments][/yellow]")
    rich.print(vars(args))

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create a path to input dataset.
    dirpath_dataset = os.path.join(args.datadir, args.category)

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

    # In training mode, run both training and test.
    if args.mode == "train":
        dataset = MVTecAD(dirpath_dataset, "train", **dataset_args)
        model.fit(dataset, args.batch_size, args.num_workers)
        model.save(os.path.join(args.outdir, "index.faiss"))

        dataset = MVTecAD(dirpath_dataset, "test", **dataset_args)
        model.score(dataset, args.n_neighbors, args.outdir, args.num_workers)

    # In test mode, run test only.
    elif args.mode == "test":
        dataset = MVTecAD(dirpath_dataset, "test", **dataset_args)
        model.load(os.path.join(args.outdir, "index.faiss"))
        model.score(dataset, args.n_neighbors, args.outdir, args.num_workers)


def main_runall(args):
    """
    Run all experiments.

    Args:
        args (argparse.Namespace): Parsed command line arguments.
    """
    rich.print("[yellow]Command line arguments:[/yellow]")
    rich.print(vars(args))

    dirpaths = [dirpath for dirpath in pathlib.Path(args.datadir).glob("*") if dirpath.is_dir()]

    for dirpath in sorted(dirpaths):

        program  = "python3 main_mvtecad.py " + ("test" if args.test_only else "train")
        datadir  = dirpath.parent
        category = dirpath.name
        model    = args.model
        repo     = args.repo
        outdir   = f"experiments/data_{model}/{category}"
        outfile  = outdir + "/log.txt"
        redirect = "" if args.no_redirect else f" > {outfile}"
        command  = f"{program} --category {category} --repo {repo} --model {model} --outdir {outdir} {redirect}"

        rich.print("[yellow]Running[/yellow]: " + command)

        # Run command.
        if not args.dryrun:
            os.makedirs(outdir, exist_ok=True)
            subprocess.run(command, shell=True)


def main_summarize(args):
    """
    Summarize experiment results.

    Args:
        args (argparse.Namespace): Parsed command line arguments.
    """
    def glob_dir(path, pattern):
        """
        Returns only directory.
        """
        for target in path.glob(pattern):
            if target.is_dir():
                yield target

    def get_value(line):
        """
        Get score value from a line string.
        """
        return float(line.strip().split(":")[-1].split()[0])

    def get_scores(filepath):
        """
        Get all scores from the given file and returns it as a dict.
        """
        scores = dict()
        for line in filepath.open():
            if   line.startswith("Total pixel-level") : scores["pixel-level"] = get_value(line)
            elif line.startswith("Total image-level") : scores["image-level"] = get_value(line)
            elif line.startswith("Feature extraction"): scores["time-featex"] = get_value(line)
            elif line.startswith("Anomaly map search"): scores["time-anmaps"] = get_value(line)
            elif line.startswith("Total infer time")  : scores["time-itotal"] = get_value(line)
        return scores

    def get_results(root):
        """
        Create a dictionaly which contains experiments results
        where the key order is `results[network][category][score]`.
        """
        results = dict()
        for dirpath in glob_dir(pathlib.Path(root), "data_*"):
            results[dirpath.name] = dict()
            for dirpath_cat in sorted(glob_dir(dirpath, "*")):
                results[dirpath.name][dirpath_cat.name] = get_scores(dirpath_cat / "log.txt")
        return results

    def print_results(results):
        """
        Print summary table to STDOUT.
        """
        networks   = list(results.keys())
        categories = list(results[networks[0]].keys())
        scores     = list(results[networks[0]][categories[0]].keys())

        # Print table (scores for each netwotks).
        for score in scores:

            header = [score] + categories
            print(",".join(header))

            # Print row (scores) for each networks.
            for network in networks:
                row = [network] + [results[network][c][score] for c in categories]
                print(",".join(map(str, row)))

    # Get results and print it.
    print_results(get_results("experiments"))


def main(args):
    """
    Entry point of this script.

    Args:
        args (argparse.Namespace): Parsed command line arguments.
    """
    if   args.mode in ["train", "test"]: main_traintest(args)
    elif args.mode in ["runall"]       : main_runall(args)
    elif args.mode in ["summ"]         : main_summarize(args)
    else                               : raise ValueError("unknown mode: " + args.mode)


if __name__ == "__main__":
    main(parse_args())
