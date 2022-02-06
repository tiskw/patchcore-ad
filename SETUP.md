Setting up
================================================================================

Installation
--------------------------------------------------------------------------------

### Using Docker (recommended)

If you don't like to pollute your development environment, it is a good idea
to run everything inside a Docker container. Our code is executable on this
Docker image. Please download the Docker image to your computer by
the following command at first:

```
docker pull tiskw/patchcore:cpu-2022-01-29
```

You can create your Docker container by the following command:

```console
cd ROOT_DIRECTORY_OF_THIS_REPO
docker run --rm -it -v `pwd`:/work -w /work -u `id -u`:`id -g` --name patchcore tiskw/patchcore:cpu-2022-01-29
```

If you need GPU support, use `tiskw/patchcore:gpu-2022-01-29` image instead,
and add `--gpus all` option to the above `docker run` command.

### Installing on your environment (easier, but pollute your development environment)

If you don't mind polluting your environment
(or you are already inside a docker container),
just run the following command for installing required packages:

```console
cd ROOT_DIRECTORY_OF_THIS_REPO
pip3 install -r requirements.txt
```

If you need GPU support, open the `requirements.txt`
and replace `faiss-cpu` to `faiss-gpu`.


Dataset
--------------------------------------------------------------------------------

Download the MVTec AD dataset from
[the official website](https://www.mvtec.com/company/research/datasets/mvtec-ad)
(or [direct link to the data file](https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938113-1629952094/mvtec_anomaly_detection.tar.xz))
and put the downloaded file (`mvtec_anomaly_detection.tar.xz`) under `data/mvtec_ad`
directory. The following is an example to download the dataset from your terminal:

```console
cd ROOT_DIRECTORY_OF_THIS_REPO
cd data/mvtec_ad
wget "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938113-1629952094/mvtec_anomaly_detection.tar.xz"
```

Then, extract the downloaded data in the `data/mvtec_ad` directory:

```console
cd ROOT_DIRECTORY_OF_THIS_REPO
cd data/mvtec_ad
tar xfJ mvtec_anomaly_detection.tar.xz
```

You've succeed to extract the dataset if the directory structure of your
`data/mvtec_ad/` is like the following:

```console
data/mvtec_ad/
|-- bottle
|   |-- ground_truth
|   |   |-- broken_large
|   |   |-- broken_small
|   |   `-- contamination
|   |-- test
|   |   |-- broken_large
|   |   |-- broken_small
|   |   |-- contamination
|   |   `-- good
|   `-- train
|       `-- good
|-- cable
|   |-- ground_truth
|   |   |-- bent_wire
|   |   |-- cable_swap
|   |   |-- combined
|   |   |-- cut_inner_insulation
|   |   |-- cut_outer_insulation
...
```

the above is the output of `tree -d --charset unicode data/mvtec_ad`
command on the authors environment.
