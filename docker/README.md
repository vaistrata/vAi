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

A demo lists the devices it finds as it starts, which is the check that the GPU reached the container.

> [!NOTE]
> **`vulkaninfo` crashes with an NVIDIA GPU** unless the container also gets the mode-setting device:
>
> ```bash
> docker run --rm -it --gpus all --device /dev/nvidia-modeset -v "$PWD/data:/opt/vai/data" vai
> ```
>
> The `vulkaninfo` in the image asks the driver for the displays (`VK_KHR_display`). `--gpus all` leaves `/dev/nvidia-modeset` out, so the query fails, and the NVIDIA driver then crashes the process. vAi makes no such query and runs without the device; AMD and Intel GPUs are not affected.

The workdir is `/opt/vai`, with the demos (`demo-*`) and the venv (`/opt/venv`) on PATH:

```bash
demo-mnist                                   # C++ demo
python python/examples/mnist/main.py \
    --weights_path data/ckpts/mnist/jroot/model.safetensors --image_path data/images/0.png
```

The pyvai suite needs pytest and torch, its numeric reference, which the image leaves out:

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cpu   # the CPU build; the default index pulls CUDA
uv pip install pytest
pytest -q
```

After editing the mounted source, `cmake --build build` rebuilds the C++ side
and `uv pip install -e .` refreshes the bindings.
