

## ⏳ Under Development: I2-HOFI

## This repository is currently under development. The final code will be released soon. Stay tuned for updates!

## Introduction
This repository implements a novel approach to Fine-Grained Visual Classification (FGVC) using Graph Neural Networks (GNNs) for advanced feature interactions. By constructing inter- and intra-region graphs, our method combines global and local features to enhance visual pattern recognition. Using shared GNNs with an attention mechanism and the APPNP algorithm, our approach optimizes information flow, boosting model efficiency and stability with residual connections. Achieving state-of-the-art results on FGVC benchmarks, this work highlights GNNs' potential in capturing complex visual details.

## Installation

### Prerequisites
Ensure you have Python 3.9+ installed on your system. You can download it from [Python's official website](https://www.python.org/downloads/).

### Virtual Environment Setup (Recommended)
Recommended installation is Anaconda(https://www.anaconda.com/download)

```bash
# Install virtualenv if you haven't installed it yet
# Create a new Conda environment
conda create --name myenv python=3.9

# Activate the Conda environment
conda activate myenv

# Install necessary packages
pip install tensorflow
pip install wandb
pip install opencv-python
pip install spektral
```

### Link to the Paper
You can access the full paper here : [Interweaving Insights: High-Order Feature Interaction for Fine-Grained Visual Recognition](https://link.springer.com/article/10.1007/s11263-024-02260-y)

### Citing paper
If you find this work useful in your research, please use the following BibTeX entry for citation.

```BibTeX
@article{sikdar2024interweaving,
  title={Interweaving Insights: High-Order Feature Interaction for Fine-Grained Visual Recognition},
  author={Sikdar, Arindam and Liu, Yonghuai and Kedarisetty, Siddhardha and Zhao, Yitian and Ahmed, Amr and Behera, Ardhendu},
  journal={International Journal of Computer Vision},
  pages={1--25},
  year={2024},
  publisher={Springer}
}
```
