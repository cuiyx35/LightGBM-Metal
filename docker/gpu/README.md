# Tiny Distroless Dockerfile for LightGBM GPU CLI-only Version

`dockerfile-cli-only-distroless.gpu` - A multi-stage build based on the `nvidia/opencl:devel-ubuntu18.04` (build) and `distroless/cc-debian10` (production) images. LightGBM (CLI-only) can be utilized in GPU and CPU modes. The resulting image size is around 15 MB.

---

# Small Dockerfile for LightGBM GPU CLI-only Version

`dockerfile-cli-only.gpu` - A multi-stage build based on the `nvidia/opencl:devel` (build) and `nvidia/opencl:runtime` (production) images. LightGBM (CLI-only) can be utilized in GPU and CPU modes. The resulting image size is around 100 MB.

---

# Dockerfile for LightGBM GPU Version with Python

`dockerfile.gpu` - A docker file with LightGBM utilizing nvidia-docker. The file is based on the `nvidia/cuda:8.0-cudnn5-devel` image.
LightGBM can be utilized in GPU and CPU modes and via Python.

## Contents

- LightGBM (cpu + gpu)
- Python (conda) + scikit-learn, notebooks, pandas, matplotlib

Running the container starts a Jupyter Notebook at `localhost:8888`.

Jupyter generates a random authentication token on startup. Retrieve the local
login URL from `docker logs lightgbm-gpu`; treat that URL as a credential and do not
share it. The notebook runs as an unprivileged user.

## Requirements

Requires docker and [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) on host machine.

## Quickstart

### Build Docker Image

```sh
mkdir lightgbm-docker
cd lightgbm-docker
wget https://raw.githubusercontent.com/lightgbm-org/LightGBM/main/docker/gpu/dockerfile.gpu
docker build -f dockerfile.gpu -t lightgbm-gpu .
```

### Run Image

```sh
nvidia-docker run --rm -d --name lightgbm-gpu -p 127.0.0.1:8888:8888 lightgbm-gpu
```

The port is published only on the host's loopback interface. No host directories
are mounted, and notebooks are removed with the container. To persist notebooks,
add `-v lightgbm-notebooks:/home/lightgbm/notebooks` for a dedicated Docker volume.
To import local data, mount only the needed directory read-only, for example
`-v "${PWD}/data:/home/lightgbm/data:ro"`. Avoid mounting the host's home directory.

### Attach with Command Line Access (if required)

```sh
docker exec -it lightgbm-gpu bash
```

### Jupyter Notebook

Open the token-bearing URL from the container logs at `http://localhost:8888`.
Keep authentication enabled; remote access should use an authenticated tunnel
instead of publishing the notebook port on all host interfaces.
