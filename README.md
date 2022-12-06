# Semantic Image Editing for Latent Diffusion

Official Implementation of the [Paper](http://arxiv.org/) **The Stable Artist: Interacting with Concepts in Diffusion Latent Space**

## Interactive Demo
An interactive demonstration is available in Colab [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ml-research/semantic-image-editing/blob/main/examples/TheStableArtist.ipynb)

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
from semdiffusers import SemanticEditPipeline
device='cuda'

pipe = SemanticEditPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
).to(device)
```

An exemplary usage of the pipeline could look like this:
```python
import torch
gen = torch.Generator(device=device)

gen.manual_seed(48)
out = pipe(prompt='a castle next to a river', generator=gen, num_images_per_prompt=1, guidance_scale=7, 
          editing_prompt=[                                    # Concepts to apply
                    'oil painting, drawing', 
                    'medieval bridge',
                    'boat on a river, boat'],
           reverse_editing_direction=[False, False, False],   # Direction of guidance
           edit_warmup_steps=[20, 10, 11],                    # Warmup period for each concept
           edit_guidance_scale=[2000, 2000, 2000],            # Guidance scale for each concept
           edit_threshold=[-0.2, -0.1, -0.1],                 # Threshold for each concept. Note that positive guidance needs negative thresholds and vice versa
           edit_weights=[1.2,1,1],                            # Weights of the individual concepts against each other
           edit_momentum_scale=0.25,                          # Momentum scale that will be added to the latent guidance
           edit_mom_beta=0.6,                                 # Momentum beta
           )
out.images[0]

```

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

