# Dream-VLA
## Installation

```bash
# Create and activate conda environment
conda create -n dreamvla python=3.10 -y
conda activate dreamvla

# Install PyTorch
# Use a comamand specific to your machine: https://pytorch.org/get-started/locally/
# pip3 install torch torchvision torchaudio
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# git clone https://github.com/DreamLM/Dream-VLX.git
# cd Dream-VLX
pip install -e .

# Install Flash Attention 2 for training (https://github.com/Dao-AILab/flash-attention)
pip install "flash-attn==2.5.5" --no-build-isolation # do the following if `import flash_attn` failed
# pip install packaging ninja
# ninja --version; echo $?  # Verify Ninja --> should return exit code "0"
# git clone https://github.com/Dao-AILab/flash-attention.git
# cd flash-attention
# python setup.py install
```

`requirements.txt` is the full used environment for reference.

## LIBERO Simulation Benchmark

### Setup
Clone and install the [LIBERO repo](https://github.com/Lifelong-Robot-Learning/LIBERO) and required packages:

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
pip install -e LIBERO
pip install -r experiments/robot/libero/libero_requirements.txt  # From vla base dir
```

(Optional, if you plan to launch training) To download the [LIBERO datasets](https://huggingface.co/datasets/openvla/modified_libero_rlds) that we used in our fine-tuning experiments, run the command below. This will download the LIBERO-Spatial, LIBERO-Object, LIBERO-Goal, and LIBERO-10 datasets in RLDS data format (~10 GB total). You can use these to fine-tune Dream-VLA or train other methods. 
Note that these are the same datasets used in the original OpenVLA project. If needed, see details on how to download the original non-RLDS datasets [here](https://github.com/openvla/openvla?tab=readme-ov-file#libero-setup).
```bash
git clone git@hf.co:datasets/openvla/modified_libero_rlds
```

### Fine-Tuning on LIBERO Datasets

First, download the LIBERO datasets as mentioned above in the Setup section above: `libero_spatial_no_noops`, `libero_object_no_noops`, `libero_goal_no_noops`, `libero_10_no_noops`. (`"_no_noops"` stands for no no-op actions, i.e., training samples with near-zero actions are filtered out).

Then, launch the fine-tuning script below:
```bash
bash vla-scripts/run-dreamvla.sh
```
You can change the fintuning objectives, currently supported objectives are bellow:
- `--use_flow_matching`: train with flow matching loss
- `--use_diffusion`: train with continuous diffusion loss
- `--use_discrete_diffusion`: train with discrete diffusion loss
- `--use_l1_regression`: train with l1 regression loss
- `--use_discrete`: train with discrete loss (i.e., one-step discrete diffusion)


### Launching LIBERO Evaluations
This is a parallel evaluation across multiple GPUs:
```bash
bash vla-scripts/parallel_eval.sh
```

## Acknowledgement
The training code is built upon [OpenVLA](https://github.com/openvla/openvla) and [OpenVLA-OFT](https://github.com/moojink/openvla-oft), we would like to thank them for the great code!