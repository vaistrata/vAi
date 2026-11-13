# vAi Docker

An image that runs vAi's C++ demos and pyvai.

## Build

The build context has to include the whole source, so run it from the repo root with `-f`:

```bash
docker build -f docker/Dockerfile -t vai .
```

## Run

Mount `data/` (MNIST ships in the repo; download the other checkpoints and mount them the same way):

```bash
# NVIDIA GPUs
docker run --rm -it --gpus all -v "$PWD/data:/opt/vai/data" vai

# AMD / Intel GPUs
docker run --rm -it --device /dev/dri -v "$PWD/data:/opt/vai/data" vai
```

NVIDIA also needs [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

The workdir is `/opt/vai`, with the demos (`demo-*`) and the venv (`/opt/venv`) on PATH:

```bash
demo-mnist                                   # C++ demo
python python/examples/mnist/main.py \
    --weights_path data/ckpts/mnist/jroot/model.safetensors --image_path data/images/0.png
pytest -q                                    # the pyvai suite
```

After editing the mounted source, `cmake --build build` rebuilds the C++ side
and `uv pip install -e .` refreshes the bindings.
