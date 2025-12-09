import argparse
import os
import csv
import pdb
import random
from collections import OrderedDict
from glob import glob
from datetime import datetime

import numpy as np
import pandas as pd
import yaml

from tqdm import tqdm

# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset, DistributedSampler
from torch.optim import lr_scheduler

import torchvision
from torchvision.utils import save_image

# Albumentations & sklearn
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose, OneOf
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# SciPy
from scipy import stats
from scipy.spatial.distance import directed_hausdorff
from scipy.stats import pearsonr, spearmanr

# Project-specific
import archs
import losses
from utils import AverageMeter, str2bool
from prov_data import *

channel_names=['DAPI', 
    'TRITC',
    'Cy5', 
    'PD-1_1:200 - Cy5', 
    'CD14 - Cy5',
    'CD4 - Cy5',
    'T-bet - Cy5', 
    'CD34 - Cy5', 
    'CD68_1:100 - TRITC', 
    'CD16 - Cy5', 
    'CD11c - Cy5',
    'CD138 - TRITC',
    'CD20 - TRITC',
    'CD3_1:1000 - Cy5',
    'CD8 - TRITC', 
    'PD-L1 - Cy5', 
    'CK_1:150 - TRITC', 
    'Ki67_1:150 - TRITC',
    'Tryptase - TRITC',
    'Actin-D - TRITC',
    'Caspase3-D - Cy5',
    'PHH3-B - Cy5',
    'Transgelin - TRITC']

ARCH_NAMES = archs.__all__
LOSS_NAMES = losses.__all__
LOSS_NAMES.append('BCEWithLogitsLoss')


def init_process_group(backend='nccl'):
    dist.init_process_group(backend=backend)
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default="mask_cell",
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--output_dir', default="./",
                        help='Output directory')

    parser.add_argument('--set', default="silver",
                        help='set to use (silver or gold)')
    parser.add_argument('--gpu_ids', nargs='+', type=int, help='A list of integers')
    parser.add_argument('--metadata', default="path_to_metadata")
    parser.add_argument('--tiling_dir', default="path_to_tiling_dir",
                        help='tiling directory')

    parser.add_argument('-b', '--batch_size', default=16, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    
    # model
    parser.add_argument('--arch', '-a', metavar='ARCH', default='gigatime',
                        )
    parser.add_argument('--input_channels', default=3, type=int,
                        help=' number of input channels')
    parser.add_argument('--num_classes', default=23, type=int,
                        help='number of classes')
    parser.add_argument('--input_w', default=556, type=int,
                        help='image width')
    parser.add_argument('--input_h', default=556, type=int,
                        help='image height')
    
    parser.add_argument('--loss', default='BCEDiceLoss',
                        help='lloss')
    
    
    parser.add_argument('--num_workers', default=12, type=int)
    
    parser.add_argument('--window_size', type=int,
                        default=256, help='window size to look into, default is 256')

    parser.add_argument('--val_sampling_prob', type=float,
                        default=0.01, help='ROI size to look into')
   

    config = parser.parse_args()
    from easydict import EasyDict as edict
    return edict(vars(config))
    return config
 


def calculate_correlations(matrix1, matrix2):
    """
    Calculate Pearson and Spearman correlation coefficients between two matrices.

    Args:
        matrix1 (np.ndarray): The first matrix.
        matrix2 (np.ndarray): The second matrix.

    Returns:
        dict: A dictionary containing Pearson and Spearman correlation coefficients.
    """
    assert matrix1.shape == matrix2.shape, "Matrices must have the same shape"
    b, c, h, w = matrix1.shape

    pearson_correlations = []
    spearman_correlations = []

    for channel in range(c):
        pearson_corrs = []
        spearman_corrs = []

        for batch in range(b):
            flat_matrix1 = matrix1[batch, channel].flatten()
            flat_matrix2 = matrix2[batch, channel].flatten()

            # Remove NaN values
            valid_indices = ~np.isnan(flat_matrix1.cpu().numpy()) & ~np.isnan(flat_matrix2.cpu().numpy())
            flat_matrix1 = flat_matrix1[valid_indices]
            flat_matrix2 = flat_matrix2[valid_indices]

            if len(flat_matrix1) > 0 and len(flat_matrix2) > 0:
                pearson_corr, _ = pearsonr(flat_matrix1.cpu().numpy(), flat_matrix2.cpu().numpy())
                spearman_corr, _ = spearmanr(flat_matrix1.cpu().numpy(), flat_matrix2.cpu().numpy())
            else:
                pearson_corr = np.nan
                spearman_corr = np.nan

            pearson_corrs.append(pearson_corr)
            spearman_corrs.append(spearman_corr)

        # Average correlations across the batch dimension, ignoring NaNs
        pearson_correlations.append(np.nanmean(pearson_corrs))
        spearman_correlations.append(np.nanmean(spearman_corrs))

    return pearson_correlations, spearman_correlations
    

def split_into_boxes(tensor, box_size):
    # Get the dimensions of the tensor
    batch_size, channels, height, width = tensor.shape
    
    # Calculate the number of boxes along each dimension
    num_boxes_y = height // box_size
    num_boxes_x = width // box_size
    
    # Split the tensor into non-overlapping boxes
    boxes = tensor.unfold(2, box_size, box_size).unfold(3, box_size, box_size)
    boxes = boxes.contiguous().view(batch_size, channels, num_boxes_y, num_boxes_x, box_size, box_size)
    
    return boxes

def count_ones(boxes):
    # Count the number of ones in each box
    return boxes.sum(dim=(4, 5))



def get_box_metrics(pred, mask, box_size):
    # Split the images into boxes
    pred_boxes = split_into_boxes(pred, box_size)
    mask_boxes = split_into_boxes(mask, box_size)
    # Count the number of ones in each box
    pred_counts = count_ones(pred_boxes)
    mask_counts = count_ones(mask_boxes)
    
    # Calculate precision and MSE for the matrices
    mse = ((pred_counts.float() - mask_counts.float()) ** 2).mean(dim=0)    
    mean_mse_per_channel = mse.mean(dim=(1,2))

    mean_mse = mse.mean().item()

    pearson, spearman = calculate_correlations(pred_counts, mask_counts)
    
    return mean_mse_per_channel, pearson, spearman 



def sample_data_loader(data_loader, config, sample_fraction=0.1, deterministic=False, what_split = "train"):


    dataset = data_loader.dataset
    total_size = len(dataset)
    sample_size = int(total_size * sample_fraction)

    if deterministic:
        sample_indices = [i for i in range(sample_size)]
    else:
        # Generate a random sample of indices
        sample_indices = random.sample(range(total_size), sample_size)
    
    # Create a subset of the dataset with the sampled indices
    subset = Subset(dataset, sample_indices)
    
    # Create a new data loader for the subset
    if what_split == "train":
        sample_loader = DataLoader(subset, batch_size=data_loader.batch_size, shuffle=True,
            num_workers=config['num_workers'],
            prefetch_factor=6,
            drop_last=True)
    else:
        sample_loader = DataLoader(subset, batch_size=data_loader.batch_size, shuffle=False,
            num_workers=config['num_workers'],
            prefetch_factor=6,
            drop_last=False)
    return sample_loader


def convert_to_csv(config, pearson_dict, name, common_channel_list):


    pear_per_class_avg = [pear.avg for pear in pearson_dict]
    pear_per_class_std = [pear.std for pear in pearson_dict]
    combined = list(zip(pear_per_class_avg, common_channel_list, pear_per_class_std))
    sorted_combined = sorted(combined, key=lambda x: x, reverse = True)
    pearson_per_class_avg, channel_names, pearson_per_class_std= zip(*sorted_combined)

    # Get today's date in YYYY-MM-DD format
    today_date = datetime.now().strftime("%Y-%m-%d")

    header = [ "Channel", "Pearson_Avg", "Pearson_Std"]

    data = {'Channel': channel_names, f'Pearson_Average {name}': pearson_per_class_avg, f'Pearson_Standard Deviation {name}': pearson_per_class_std
            }
    df = pd.DataFrame(data)
    df.to_csv(config['output_dir'] + "models/" + config['name'] + "/" + config['set'] + f'/{name}_per_channel_{today_date}_test_results.csv', index=False)





def small_tile_preds(big_image,output_image, model,  window_size=256):
    for i in range(0, big_image.shape[2], window_size):
        for j in range(0, big_image.shape[3], window_size):
                # Extract the window from the small image
                window = big_image[:,:, i:i + window_size, j:j + window_size].cuda()
                
                output = model(window)
                output_image[:,:,i:i + window_size, j:j + window_size] = output

    return output_image
 
def validate(config, val_loader, model, criterion,  common_channel_list):
    avg_meters = {'loss': AverageMeter(),
                  'pearson': AverageMeter()}


    pearson_per_class_meters = [AverageMeter() for _ in range(config['num_classes'])]
    # switch to evaluate mode


    window_size = config['window_size']
    model.eval()
    count = 0


    with torch.no_grad():
        pbar = tqdm.tqdm(total=len(val_loader))
        
        for data in val_loader:
            count+=1
            input, target_g, name = data
            

            downsampled_image = F.interpolate(target_g, scale_factor=1/8, mode='bilinear', align_corners=False)
            target = F.interpolate(downsampled_image, size=(512,512), mode='bilinear', align_corners=False)

            target = target.cuda()

            output_image = torch.zeros_like(target).cuda()

            output_image = small_tile_preds(input,output_image, model, window_size)

            output_image = output_image > 0.5
            output_image = output_image.float()
            loss = criterion(output_image, target)

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg)
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
            box_size = 8

            _, pearson, spearman = get_box_metrics(output_image, target, box_size)


            for class_idx, value in enumerate(pearson):
                if not np.isnan(value):
                    pearson_per_class_meters[class_idx].update(value, target.size(0))


    postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg)
            ]+[(channel_name, pearson_per_class_meters[channel_idx].avg) for channel_idx, channel_name in enumerate(common_channel_list)]
            )
    pbar.set_postfix(postfix)
    pbar.close()


    convert_to_csv(config,  pearson_per_class_meters, "Results", common_channel_list)

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('pearson', avg_meters['pearson'].avg)]\
                            +[(f'{channel_name}_pearson', pearson_per_class_meters[channel_idx].avg) for channel_idx, channel_name in enumerate(common_channel_list)]\
            )

def print_logs(log, exclude_keys=[]):
    for key, value in log.items():
        if key not in exclude_keys:
            print(f"{key}: {value:.4f}")

def main():
    
    common_channel_list=['DAPI', 
    'TRITC',
    'Cy5', 
    'PD-1_1:200', 
    'CD14',
    'CD4',
    'T-bet', 
    'CD34', 
    'CD68_1:100', 
    'CD16', 
    'CD11c',
    'CD138',
    'CD20',
    'CD3_1:1000',
    'CD8', 
    'PD-L1', 
    'CK_1:150', 
    'Ki67_1:150',
    'Tryptase',
    'Actin-D',
    'Caspase3-D',
    'PHH3-B',
    'Transgelin']

    config = vars(parse_args())


    os.makedirs(config['output_dir'] + 'models/%s' % config['name'], exist_ok=True)
    os.makedirs(config['output_dir'] + "models/" +config['name'] +"/" + config['set'] , exist_ok=True)
    print('-' * 20)
    for key in config:
        print('%s: %s' % (key, config[key]))
    print('-' * 20)

    with open(config['output_dir'] +'models/%s/config.yml' % config['name'], 'w') as f:
        yaml.dump(config, f)

    # define loss function (criterion)
    if config['loss'] == 'MSELoss':
        criterion = nn.MSELoss().cuda()
    elif config['loss'] == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().cuda()
    elif config['loss'] == 'BCEDiceLoss':
        criterion = losses.BCEDiceLoss().cuda()
    else:
        criterion = losses.__dict__[config['loss']]().cuda()

    cudnn.benchmark = False

    # create model
    print("=> creating model %s" % config['arch'])
    model = archs.__dict__[config['arch']](config['num_classes'],
                                            config['input_channels']).cuda()

    # Load it using the huggingface model card
    from huggingface_hub import snapshot_download

    repo_id = "prov-gigatime/GigaTIME"

    # Download the repo snapshot 
    local_dir = snapshot_download(repo_id=repo_id)

    weights_path = os.path.join(local_dir, "model.pth")
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model = torch.nn.DataParallel(model)
    params = filter(lambda p: p.requires_grad, model.parameters())
    

 
    
    import albumentations as geometric

    val_transform = Compose([
        geometric.Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])
        
        
    metadata = pd.read_csv(config["metadata"])
    tilting_dir = Path(config["tiling_dir"])
    tile_pair_df=generate_tile_pair_df(metadata=metadata, tiling_dir=tilting_dir)

    # Filter DataFrame (remove empty patches / remove patches from pairs where registration was not successful.)
    tile_pair_df_filtered = tile_pair_df[tile_pair_df.apply(
        lambda x:
            # Check conditions for filtering: These are decided based on manual checks as well as discussions with biologists
            # 1. Black ratio of comet image is less than 0.3
            # 2. Variance of comet image is greater than 200
            # 3. Black ratio of HE image is less than 0.3
            # 4. Variance of HE image is greater than 200
            # 5. Registration parameter for the pair is None (indicating unsuccessful registration)
            ((x["img_comet_black_ratio"] < 0.3) &
             (x["img_comet_variance"] > 200) &
             (x["img_he_black_ratio"] < 0.3) &
             (x["img_he_variance"] > 200)) , axis=1
    )]



    dir_names = tile_pair_df_filtered["dir_name"].unique()
    segment_metric_dict = {}

    # Load the segment metrics into a dictionary
    for dir_name in dir_names:
        with open(os.path.join(dir_name, "segment_metric.json"), "r") as f:
            segment_metric_list = json.load(f)
        segment_metric_dict[dir_name] = segment_metric_list

    # Prepare a dictionary to store the new columns
    new_columns = {col: [] for col in next(iter(segment_metric_dict[dir_names[0]].values())).keys()}

    # Populate the dictionary with the corresponding values
    for _, row in tile_pair_df_filtered.iterrows():
        metrics = segment_metric_dict[row["dir_name"]][row["pair_name"]]
        for key, value in metrics.items():
            new_columns[key].append(value)

    # Add the new columns to the DataFrame
    for key, values in new_columns.items():
        tile_pair_df_filtered[key] = values

    # Now tile_pair_df_filtered will have the new columns added from the segment_metric_dict
        
    tile_pair_df_filtered_dicefilter=tile_pair_df_filtered[tile_pair_df_filtered["dice"]>0.2]
    print(tile_pair_df_filtered_dicefilter.shape)

    test_dataset = HECOMETDataset_roi(
        all_tile_pair=tile_pair_df,
        tile_pair_df=tile_pair_df_filtered_dicefilter,
        transform=val_transform,
        dir_path = config["tiling_dir"],
        window_size = config["window_size"],
        split="test",
        standard = config["set"],
        mask_noncell=True,
        cell_mask_label=True,
    )
 
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        prefetch_factor=4,
        drop_last=False)

    log = OrderedDict([
        ('epoch', []),
        ('lr', []),
        ('loss', []),
        ('pearson', []),
        ('val_loss', []),
        ('val_pearson', []),

    ] + [(channel_name, []) for channel_name in common_channel_list] + 
    [('val_' + channel_name, []) for channel_name in common_channel_list])

    
    test_loader = sample_data_loader(test_loader, config,config['val_sampling_prob'], deterministic=True, what_split = "valid")

    test_log_silver = validate(config, test_loader, model, criterion, common_channel_list=common_channel_list)
    print_logs(test_log_silver)

    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()