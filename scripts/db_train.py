import argparse
import os
import random
from collections import OrderedDict
from glob import glob

import numpy as np
import pandas as pd
import yaml
import tqdm
from scipy.stats import pearsonr, spearmanr

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Subset

import torchvision
from torchvision.utils import save_image

from albumentations.augmentations import transforms
from albumentations.core.composition import Compose, OneOf
from sklearn.model_selection import train_test_split

from easydict import EasyDict as edict
from torch.optim import lr_scheduler

# project-specific imports
from archs import gigatime
from losses import *
from utils import *
from prov_data import *

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default="mask_cell",
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--output_dir', default="./",
                        help='Output directory')

    parser.add_argument('--gpu_ids', nargs='+', type=int, help='A list of integers')
    parser.add_argument('--metadata', default="path_to_metadata",
                        help='Output directory')
    parser.add_argument('--tiling_dir', default="path_to_tiling_dir",
                        help='Output directory')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=16, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    
    # model
    parser.add_argument('--arch', '-a', metavar='ARCH', default='gigatime',
                        help='no_help')
    parser.add_argument('--input_channels', default=3, type=int,
                        help='input channels')
    parser.add_argument('--num_classes', default=23, type=int,
                        help='number of classes')
    parser.add_argument('--input_w', default=556, type=int,
                        help='image width')
    parser.add_argument('--input_h', default=556, type=int,
                        help='image height')
    
    # loss
    parser.add_argument('--loss', default='BCEDiceLoss',
                        help='lloss')

    # optimizer
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                        ' | '.join(['Adam', 'SGD']) +
                        ' (default: Adam)')
    parser.add_argument('--lr', '--learning_rate', default=1e-3, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')

    # scheduler
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
    parser.add_argument('--min_lr', default=1e-5, type=float,
                        help='minimum learning rate')
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--gamma', default=2/3, type=float)
    parser.add_argument('--early_stopping', default=-1, type=int,
                        metavar='N', help='early stopping (default: -1)')
    
    parser.add_argument('--num_workers', default=12, type=int)

    parser.add_argument('--window_size', type=int,
                        default=256)
    parser.add_argument('--sampling_prob', type=float,
                        default=1, help='training data loader sampling probability for easy debugging, set 1 for full training')
    parser.add_argument('--val_sampling_prob', type=float,
                        default=1, help='validation data loader sampling probability for easy debugging, set 1 for full validation')
    parser.add_argument('--crop', type=str2bool, default=False)
    parser.add_argument('--sigmoid', type=str2bool, default=True)


    config = parser.parse_args()
    from easydict import EasyDict as edict
    return edict(vars(config))
    return config

mean = torch.tensor([0.485, 0.456, 0.406]).cuda()
std = torch.tensor([0.229, 0.224, 0.225]).cuda()


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
            valid_indices = ~np.isnan(flat_matrix1.detach().cpu().numpy()) & ~np.isnan(flat_matrix2.detach().cpu().numpy())
            flat_matrix1 = flat_matrix1[valid_indices]
            flat_matrix2 = flat_matrix2[valid_indices]

            if len(flat_matrix1) > 0 and len(flat_matrix2) > 0:
                pearson_corr, _ = pearsonr(flat_matrix1.detach().cpu().numpy(), flat_matrix2.detach().cpu().numpy())
                spearman_corr, _ = spearmanr(flat_matrix1.detach().cpu().numpy(), flat_matrix2.detach().cpu().numpy())
            else:
                pearson_corr = np.nan
                spearman_corr = np.nan

            pearson_corrs.append(pearson_corr)
            spearman_corrs.append(spearman_corr)

        # Average correlations across the batch dimension
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





def sample_data_loader(data_loader, config, sample_fraction=0.1, deterministic=False, what_split="train"):

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

def denormalize(tensor, mean, std):
    mean = mean[None, :, None, None]
    std = std[None, :, None, None]
    return tensor * std + mean
def train(config, train_loader, model, criterion, optimizer):
    # Initialize average meters to track loss and Pearson correlation metrics
    avg_meters = {'loss': AverageMeter(), 'pearson': AverageMeter()}
    pearson_per_class_meters = [AverageMeter() for _ in range(config['num_classes'])]
    window_size = config['window_size']
    
    # Set model to training mode
    model.train()

    # Initialize progress bar for training loop
    pbar = tqdm.tqdm(total=len(train_loader))
    for input, target, name in train_loader:
        # Downsample target by factor of 8, then resize to input dimensions to make the target coarse to discount for any pixel level registration error
        downsampled_image = F.interpolate(target, scale_factor=1/8, mode='bilinear', align_corners=False)
        target = F.interpolate(downsampled_image, size=(config["input_h"],config["input_h"]), mode='bilinear', align_corners=False)
        target = target.cuda()
        
        # Forward pass through model
        output_image = model(input.cuda()).cuda()

        # Calculate loss between predicted and target images
        loss = criterion(output_image, target)
        
        # Backpropagation and parameter update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate pearson metrics
        _, pearson, _ = get_box_metrics(output_image, target, box_size=8)

        
        # Update per-class Pearson meters
        for class_idx, pearson_value in enumerate(pearson):
            pearson_per_class_meters[class_idx].update(pearson_value, input.size(0))

        # Update average meters with current batch metrics
        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['pearson'].update(np.nanmean(pearson), input.size(0))

        # Update progress bar with current metrics
        pbar.set_postfix({'loss': avg_meters['loss'].avg, 'pearson': avg_meters['pearson'].avg})
        pbar.update(1)
    pbar.close()

    # Return ordered dictionary with training metrics
    return OrderedDict([('loss', avg_meters['loss'].avg), ('pearson', avg_meters['pearson'].avg)] +
                       [(f'class_{i}', m.avg) for i, m in enumerate(pearson_per_class_meters)])

def validate(config, val_loader, model, criterion):
    # Initialize average meters to track validation loss and Pearson correlation metrics
    avg_meters = {'loss': AverageMeter(), 'pearson': AverageMeter()}
    pearson_per_class_meters = [AverageMeter() for _ in range(config['num_classes'])]
    window_size = config['window_size']
    
    # Set model to evaluation mode (disables dropout, batch norm updates)
    model.eval()

    # Disable gradient computation for validation (saves memory and computation)
    with torch.no_grad():
        # Initialize progress bar for validation loop
        pbar = tqdm.tqdm(total=len(val_loader))
        for input, target, name in val_loader:
            # Downsample target by factor of 8, then resize to input dimensions
            downsampled_image = F.interpolate(target, scale_factor=1/8, mode='bilinear', align_corners=False)
            target = F.interpolate(downsampled_image, size=(config["input_h"],config["input_h"]), mode='bilinear', align_corners=False)
            target = target.cuda()
            
            # Forward pass through model
            output_image = model(input.cuda()).cuda()

            # Calculate validation loss
            loss = criterion(output_image, target)

            # Calculate  metrics 
            _, pearson, _ = get_box_metrics(output_image, target, box_size=8)

            # Update per-class Pearson meters
            for class_idx, pearson_value in enumerate(pearson):
                pearson_per_class_meters[class_idx].update(pearson_value, input.size(0))

            # Update average meters with current batch metrics
            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['pearson'].update(np.nanmean(pearson), input.size(0))

            # Update progress bar with current metrics
            pbar.set_postfix({'loss': avg_meters['loss'].avg, 'pearson': avg_meters['pearson'].avg})
            pbar.update(1)
        pbar.close()

    # Return ordered dictionary with validation metrics
    return OrderedDict([('loss', avg_meters['loss'].avg), ('pearson', avg_meters['pearson'].avg)] +
                       [(f'class_{i}', m.avg) for i, m in enumerate(pearson_per_class_meters)])

def main():
    
    # Define a list of channel names used in the dataset in the order they are found in the COMET files
    # TRITC and Cy5 are background channels and will be ignored evaluation
    # The other TIME markers are the ones we will evaluate and they are subtracted from the background channels during data acquisition and preprocessing itself
    common_channel_list=['DAPI', 
    'TRITC', # background channel
    'Cy5', # background channel
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

    config = vars(parse_args())

    os.makedirs(config['output_dir'] + 'models/%s' % config['name'], exist_ok=True)
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
        criterion = BCEDiceLoss().cuda()
    else:
        criterion = losses.__dict__[config['loss']]().cuda()

    cudnn.benchmark = False

    # create model
    print("=> creating model %s" % config['arch'])

    model = gigatime(num_classes=config['num_classes'],

                                            sigmoid=config["sigmoid"],

                                            loss_type=config["loss"],

                                            input_channels=config['input_channels'],).cuda()


    if len(config["gpu_ids"]) > 1:
        device_ids = config["gpu_ids"]
        model = nn.DataParallel(model, device_ids=device_ids)
        print("using multiple GPUs", config["gpu_ids"])

    params = filter(lambda p: p.requires_grad, model.parameters())
    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(
            params, lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(params, lr=config['lr'], momentum=config['momentum'],
                              nesterov=config['nesterov'], weight_decay=config['weight_decay'])
    else:
        raise NotImplementedError

    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'], patience=config['patience'],
                                                   verbose=1, min_lr=config['min_lr'])
    elif config['scheduler'] == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in config['milestones'].split(',')], gamma=config['gamma'])
    elif config['scheduler'] == 'ConstantLR':
        scheduler = None
    else:
        raise NotImplementedError


    import albumentations as geometric
    if config['crop']:
        train_transform = Compose([
            geometric.RandomRotate90(),
            geometric.Flip(),
            OneOf([
                transforms.HueSaturationValue(),
                transforms.RandomBrightnessContrast(brightness_limit=0, contrast_limit=0.2),
                transforms.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0),
            ], p=1),
            geometric.RandomCrop(config['input_h'], config['input_w']),
            transforms.Normalize()
        ],
            is_check_shapes=False)

        val_transform = Compose([
            geometric.RandomCrop(config['input_h'], config['input_w']),
            transforms.Normalize()
        ],
            is_check_shapes=False)

    else:
        train_transform = Compose([
            geometric.RandomRotate90(),
            geometric.Flip(),
            OneOf([
                transforms.HueSaturationValue(),
                transforms.RandomBrightnessContrast(brightness_limit=0, contrast_limit=0.2),
                transforms.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0),
            ], p=1),
            geometric.Resize(config['input_h'], config['input_w']),
            transforms.Normalize()
        ],
            is_check_shapes=False)

        val_transform = Compose([
            geometric.Resize(config['input_h'], config['input_w']),
            transforms.Normalize()
        ],
            is_check_shapes=False)
        
    metadata = pd.read_csv(config["metadata"])
    # Define the tiling directory path from the configuration
    tiliting_dir = Path(config["tiling_dir"])

    # Generate a DataFrame containing tile pairs based on metadata and the tiling directory
    tile_pair_df = generate_tile_pair_df(metadata=metadata, tiling_dir=tiliting_dir)
    # Filter the DataFrame to remove empty patches and patches from pairs where registration was not successful
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
        # Open the JSON file containing segmentation metrics for the current directory
        with open(os.path.join(dir_name, "segment_metric.json"), "r") as f:
            segment_metric_list = json.load(f)
        segment_metric_dict[dir_name] = segment_metric_list


    new_columns = {col: [] for col in next(iter(segment_metric_dict[dir_names[0]].values())).keys()}

    for _, row in tile_pair_df_filtered.iterrows():
        metrics = segment_metric_dict[row["dir_name"]][row["pair_name"]]
        for key, value in metrics.items():
            new_columns[key].append(value)

    # Add the new columns to the DataFrame
    # This step integrates the segmentation metrics into the main DataFrame for further analysis
    for key, values in new_columns.items():
        tile_pair_df_filtered[key] = values

    # Filter the DataFrame to retain only rows where the "dice" metric is greater than 0.2
    # This step ensures that only high-quality segmentation results between DAPI and stardist outputs are included in the dataset
    tile_pair_df_filtered_dicefilter = tile_pair_df_filtered[tile_pair_df_filtered["dice"] > 0.2]

    train_dataset = HECOMETDataset_roi(
        all_tile_pair=tile_pair_df,
        tile_pair_df=tile_pair_df_filtered_dicefilter,
        transform=train_transform,
        dir_path = config["tiling_dir"],
        window_size = config["window_size"],
        split="train",
        mask_noncell=True,
        cell_mask_label=True,
    )    

    val_dataset = HECOMETDataset_roi(
        all_tile_pair=tile_pair_df,
        tile_pair_df=tile_pair_df_filtered_dicefilter,
        transform=val_transform,
        dir_path = config["tiling_dir"],
        window_size = config["window_size"],
        split="valid",
        standard = "silver",
        mask_noncell=True,
        cell_mask_label=True,
    )    


    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        prefetch_factor=6,
        drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        prefetch_factor=6,
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

    val_loader = sample_data_loader(val_loader, config, config['val_sampling_prob'], deterministic=True, what_split = "valid")

    best_pearson = 0
    trigger = 0
    for epoch in range(config['epochs']):
        print('Epoch [%d/%d]' % (epoch, config['epochs']))


        train_log = train(config, train_loader, model, criterion, optimizer)
        # evaluate on validation set
        val_log = validate(config, val_loader, model, criterion)

        if epoch%10==0: ##this is for saving intermediate results and visualization

            input = val_log['input'].cuda()
            target = val_log['target']
            output = val_log['output']
            input = denormalize(input, mean, std)
            grid = torchvision.utils.make_grid(input, nrow=1)
            save_image(grid, config['output_dir'] + "models/" + config['name'] +'/HE_image.png')
            grid = torchvision.utils.make_grid(target[:,0,:,:].unsqueeze(1), nrow=1)
            save_image(grid, config['output_dir'] + "models/" +config['name'] +'/target.png')
            grid = torchvision.utils.make_grid(output[:,0,:,:].unsqueeze(1), nrow=1)
            save_image(grid, config['output_dir'] + "models/" +config['name'] + '/output.png')


        if config['scheduler'] == 'CosineAnnealingLR':
            scheduler.step()
        elif config['scheduler'] == 'ReduceLROnPlateau':
            scheduler.step(val_log['loss'])



        trigger += 1

        if val_log['pearson'] > best_pearson: #save best model
            torch.save(model.module.state_dict(), config['output_dir'] +'models/%s/model.pth' %
                       config['name'])
            best_pearson = val_log['pearson']
            print("=> saved best model")

            input = val_log['input'].cuda()
            target = val_log['target']
            output = val_log['output']
            input = denormalize(input, mean, std)
            grid = torchvision.utils.make_grid(input, nrow=1)
            save_image(grid, config['output_dir'] + "models/" + config['name'] +'/HE_image_best.png')
            grid = torchvision.utils.make_grid(target[:,0,:,:].unsqueeze(1), nrow=1)
            save_image(grid, config['output_dir'] + "models/" +config['name'] +'/target_best.png')
            grid = torchvision.utils.make_grid(output[:,0,:,:].unsqueeze(1), nrow=1)
            save_image(grid, config['output_dir'] + "models/" +config['name'] + '/output_best.png')

        # early stopping
        if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
            print("=> early stopping")
            break

        torch.cuda.empty_cache()


if __name__ == "__main__":
    t =main()