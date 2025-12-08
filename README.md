# GigaTIME: Multimodal AI generates virtual population for tumor microenvironment modeling (Cell)

<div align="center">

[![Paper](https://img.shields.io/badge/Paper-Cell-red.svg)](https://www.cell.com/)
[![Model](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-yellow)](https://huggingface.co/prov-gigatime/GigaTIME)
[![License](https://img.shields.io/badge/License-Research%20Only-blue.svg)](#license-notice)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)
[![Microsoft](https://img.shields.io/badge/Microsoft-Research-00A4EF.svg?logo=microsoft)](https://www.microsoft.com/en-us/research/)


*Official implementation of GigaTIME*

[📄 Paper](https://www.cell.com/) • [🤗 Model Card](https://huggingface.co/prov-gigatime/GigaTIME) 

</div>

## Environment Setup

We recommend using Conda for environment management. The codebase has been tested with Python 3.11 using A100 GPUs for optimal reproducibility. Before creating the environment, ensure that the `torch` version specified in `environment.yml` matches your GPU and CUDA driver setup.

To set up the environment, run:

```bash
conda env create -f environment.yml
```

This will create a Conda environment named `gigatime`. Activate it with:

```bash
conda activate gigatime
```

## Data 

A set of 50 paired H&E and mIF patches from the test set is available for evaluation. Download the sample data from [Dropbox](https://www.dropbox.com/scl/fi/8ampg43fs2yowt9y6vvr1/sample_test_data.zip?rlkey=bkg4w183qnvkh2dudqy3d8lsg&st=j2l463ug&dl=0).

After downloading, unzip the folder and place it in the `data` directory:

```bash
unzip sample_test_data.zip -d ./data/
```

Make sure the extracted folder are located in `./data/`.

## Pre-trained Model

Load directly from Hugging Face

Model card available in [HuggingFace](https://huggingface.co/prov-gigatime/GigaTIME) 

```
from huggingface_hub import hf_hub_download
import torch

# Download model weights
weights_path = hf_hub_download(
    repo_id="prov-gigatime/GigaTIME",
    filename="model.pth"  
)

# Load model
state_dict = torch.load(weights_path, map_location='cpu')
model.load_state_dict(state_dict)
```

## Tutorials

- **Inference Tutorial:** 

Learn how to load the model and run predictions on sample patches:


    ```
    scripts/gigatime_testing.ipynb
    ```


- **Training Tutorial:** 

Understand the training workflow with a one-epoch demo:


    ```
    scripts/gigatime_training.ipynb
    ```

## Training GigaTIME cross-modal translator


We also release the script needed to train the GigaTIME model here. 

To train the model:

```
python scripts/db_train.py --arch gigatime   --tiling_dir "gigatime_training_path"  --window_size 256       --batch_size 32     --sampling_prob 1     --name GigaTIME_model    --output_dir "Output_Directory"    --epoch 300 --input_h 512 --input_w 512 --lr 0.001 --loss BCEDiceLoss --val_sampling_prob 1 --num_workers 12 --gpu_ids 0 1 2 3 4 5 6 7 --crop True --metadata "Gigatime metadata file"
```

## Model Uses

### Intended Use
The data, code, and model checkpoints are intended to be used solely for (I) future research on pathology foundation models and (II) reproducibility of the experimental results reported in the reference paper. The data, code, and model checkpoints are not intended to be used in clinical care or for any clinical decision-making purposes.

### Primary Intended Use
The primary intended use is to support AI researchers reproducing and building on top of this work. GigaPath should be helpful for exploring pre-training, and encoding of digital pathology slides data.

### Out-of-Scope Use
Any deployed use case of the model --- commercial or otherwise --- is out of scope. Although we evaluated the models using a broad set of publicly-available research benchmarks, the models and evaluations are intended for research use only and not intended for deployed use cases.

## License Notice

The model is not intended or made available for clinical use as a medical device, clinical support, diagnostic tool, or other technology intended to be used in the diagnosis, cure, mitigation, treatment, or prevention of disease or other conditions. The model is not designed or intended to be a substitute for professional medical advice, diagnosis, treatment, or judgment and should not be used as such. All users are responsible for reviewing the output of the developed model to determine whether the model meets the user’s needs and for validating and evaluating the model before any clinical use.

## Citation

```
@article{valanarasu2025gigatime,
  title={Multimodal AI generates virtual population for tumor microenvironment modeling},
  author={Valanarasu, Jeya Maria Jose and Xu, Hanwen and Usuyama, Naoto and Kim, Chanwoo and Wong, Cliff and Argaw, Peniel and Ben Shimol, Racheli and Crabtree, Angela and Matlock, Kevin and Bartlett, Alexandra Q. and Bagga, Jaspreet and Gu, Yu and Zhang, Sheng and Naumann, Tristan and Fox, Bernard A. and Wright, Bill and Robicsek, Ari and Piening, Brian and Bifulco, Carlo and Wang, Sheng and Poon, Hoifung},
  journal={Cell},
  year={2025},
  publisher={Cell Press}
}
```
