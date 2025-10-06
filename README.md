# Multimodal AI generates virtual population for tumor microenvironment modeling (Cell)

### Official GigaTIME Codebase

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

The pre-trained GigaTIME model can be downloaded [here](https://www.dropbox.com/scl/fi/phg4as7s8ayemwg64r27a/model.pth?rlkey=1n9skwtfduq2qdj6c6uor0myv&st=t209twfl&dl=0).

After downloading, move the `model.pth` file to the `models` directory:

```bash
mv model.pth ./models/
```

## Tutorials

- **Inference Tutorial:** Demonstrates loading the model, preparing data, and running inference on sample patches.

    ```
    scripts/gigatime_testing.ipynb
    ```

- **Training Tutorial:** Provides an overview of the training workflow and demonstrates training for one epoch.

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