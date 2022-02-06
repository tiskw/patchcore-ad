Docker images
================================================================================

This directory contains files for building Docker images used in
this repository. All images are available from
[Dockerhub](https://hub.docker.com/r/tiskw/patchcore).

Build docker images
--------------------------------------------------------------------------------

### Docker image for CPU

```console
cd ROOT_DIRECTORY_OF_THIS_REPO/docker/cpu
docker build -t `date +"tiskw/patchcore:cpu-%Y-%m-%d"` .
```

### Docker image for GPU

```console
cd ROOT_DIRECTORY_OF_THIS_REPO/docker/gpu
docker build -t `date +"tiskw/patchcore:gpu-%Y-%m-%d"` .
```
