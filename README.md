# Semantic Image Editing for Latent Diffusion

Official Implementation of the [Paper](http://arxiv.org/) **The Stable Artist: Interacting with Concepts in Diffusion Latent Space**

## Interactive Demo
An interactive demonstration is available in Colab [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)]([https://colab.research.google.com/](https://colab.research.google.com/)).

## Installation
You can either clone the repository and install it locally by running

```cmd
git clone https://github.com/ml-research/semantic-image-editing.git
cd ./semantic-image-editing
pip install .
```
or install it directly from git
```cmd
pip install git+https://github.com/ml-research/semantic-image-editing.git
```

## Usage
This repository provides a new diffusion pipeline supporting semantic image editing based on the [diffusers](https://github.com/huggingface/diffusers) library.
The ```SemanticEditPipeline``` extends the ```StableDiffusionPipeline``` and can therefore be loaded from a stable diffusion checkpoint like shown below.


```python
from sem_diffusers import SemanticEditPipeline
device='cuda'

pipe = SemanticEditPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
).to(device)
```
###TODO:
Example

## Citation
If you like or use our work please cite us:
```bibtex
@article{brack2022Stable,
      title={The Stable Artist: Interacting with Concepts in Diffusion Latent Space}, 
      author={Manuel Brack and Patrick Schramowski and Felix Friedrich and Kristian Kersting},
      year={2022},
      journal={arXiv preprint arXiv:XXX.XXX}
}
```

