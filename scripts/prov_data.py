import random
import pickle
import os
from pathlib import Path
import subprocess
from collections import Counter
import traceback
import tempfile
import time
import json
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from skimage.segmentation import find_boundaries
import pandas as pd
import tqdm
import glob
from PIL import Image
import numpy as np
import shutil
import os
import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split

import albumentations as A
import gzip

import logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("py4j").setLevel(logging.ERROR)


common_channel_list=['DAPI', 'TRITC', 'Cy5', 'Caspase 3 - Cy5', 'Transgelin - TRITC', 'T-bet - Cy5', 'Actin - TRITC', 'CD4 - Cy5', 'CD68 - TRITC', 'CD34 - Cy5', 'Ki67 - TRITC', 'CD11c - Cy5', 'Tryptase - TRITC', 'CD14 - Cy5', 'CK - TRITC', 'CD3 - Cy5', 'CD8 - TRITC', 'CD66b - Cy5', 'CD20 - TRITC', 'PD-1 - Cy5', 'CD138 - TRITC', 'PDL-1 - Cy5', 'CD16 - Cy5']


def update_dict_with_key_check(target_dict, source_dict, key_list=None):
    """
    Update the target dictionary with values from the source dictionary.
    Ensures no duplicate keys are added to the target dictionary.

    Parameters:
    target_dict (dict): The dictionary to update.
    source_dict (dict): The dictionary to copy values from.
    key_list (list, optional): List of keys to copy. Defaults to all keys in source_dict.
    """
    if key_list is None:
        key_list = source_dict.keys()
    for key in key_list:
        assert key not in target_dict.keys()
        target_dict[key] = source_dict[key]


def generate_tile_pair_df(metadata, tiling_dir):
    """
    Generate a DataFrame containing image metric information for each tile pair.

    Parameters:
    metadata (pd.DataFrame): DataFrame containing metadata information.
    tiling_dir (Path): Directory containing tiling results.

    Returns:
    pd.DataFrame: DataFrame containing image metrics for each tile pair.
    """
    img_metric_dict_all = {}
    for idx, row in metadata.iterrows():  
        img_metric_dict_new = {}
        
        # Paths to JSON files containing image statistics and comet metadata
        img_stats_path = tiling_dir / f'{row["tiff_filename"]}_and_{row["he_filename"].replace("#", "")}' / "img_statistics.json"
        comet_metadata_path = tiling_dir / f'{row["tiff_filename"]}_and_{row["he_filename"].replace("#", "")}' / "comet_metadata.json"

        # Load JSON data
        if not os.path.exists(img_stats_path):
            print(f"File not found: {img_stats_path}")
            continue
        with open(img_stats_path, "r") as f:
            img_metric_dict = json.load(f)
        with open(comet_metadata_path, 'r') as fp:
            comet_metadata = json.load(fp)        

        # Update img_metric with metadata and comet metadata
        for pair_name, img_metric in img_metric_dict.items():
            img_metric["pair_name"] = pair_name
            img_metric["dir_name"] = str(img_stats_path.parent)

            update_dict_with_key_check(img_metric, row.to_dict(), key_list=[
                'slide_deid', 
                'tiff_filename', 'he_filename'
            ])
            update_dict_with_key_check(img_metric, comet_metadata, key_list=[
                "channel_names", "n_channels", "pixel_physical_size_xyu", 
                "tiling_num_tiles", "tiling_patch_size_um", "tiling_overlap"
            ])

            img_metric_dict_new[f'{row["tiff_filename"]}_and_{row["he_filename"].replace("#", "")}_{pair_name}'] = img_metric

        img_metric_dict_all.update(img_metric_dict_new)

    img_df = pd.DataFrame(img_metric_dict_all).T
    return img_df




def unpack_and_load(file_path):
    with gzip.open(file_path, 'rb') as f:
        data = pickle.load(f)

    comet_array_binary_packed = data["comet_array_binary"] # Shape: (M, N, K) where K is the packed dimension
    original_shape = data["original_shape"]  # Shape: (M, N, C)
    original_last_dim = data["original_last_dim"] # Scalar, original last dimension 
    
    # Unpack the binary comet array
    comet_array_binary = np.unpackbits(comet_array_binary_packed, axis=-1) # Shape: (M, N, K * 8)
    
    # Slice off the padding to restore the original shape
    comet_array_binary = comet_array_binary[..., :original_last_dim].reshape(original_shape) # Shape: (M, N, C)

    # No need to revert label data types
    data["comet_array_binary"] = comet_array_binary

    return data



def get_image_roi(x, y, images_dict, dir_path, pair):
    key = f"{x}_{y}_556_556"
    fol_name = pair["dir_name"].split("/")[-1]

    he_image_path = os.path.join(dir_path,fol_name , key+ "_he.png")
    images = np.array(Image.open(he_image_path).convert("RGB"))

    return images



def get_image(he_image_path):

    images = np.array(Image.open(he_image_path).convert("RGB"))

    return images



            
def image_reader(image_path, transform=None):
    # Open the image file
    img = np.array(Image.open(image_path).convert('RGB'))
    
    # Apply the transform if provided
    if transform:
        augmented =transform(image=img, mask=img)
        img = augmented['image']
        img = img.astype('float32')
        img = img.transpose(2, 0, 1)  # Convert to CHW format
    
    # Convert the image to a PyTorch tensor
    # image_tensor = transforms.ToTensor()(image)
    
    return img

class HECOMETDataset_roi(torch.utils.data.Dataset):
    def __init__(self,all_tile_pair, tile_pair_df, mask_noncell, transform, 
                 cell_mask_label, dir_path, window_size, 
                 split="train", standard = "all"):
        self.all_tile_pair = all_tile_pair
        self.tile_pair_df = tile_pair_df
        self.mask_noncell=mask_noncell
        self.transform = transform
        self.cell_mask_label=cell_mask_label
        self.split=split
        self.standard=standard
        self.dir_path = dir_path
        self.window_size = window_size

        # Split into train, validation, and test sets
        if split == "full":
            self.tile_pair_df = self.tile_pair_df
        else:
            
            train_dir, temp_dir = train_test_split(self.tile_pair_df["dir_name"].unique(), test_size=0.4, random_state=42)
            valid_dir, test_dir = train_test_split(temp_dir, test_size=0.5, random_state=42)
            if split == "train":
                self.tile_pair_df = self.tile_pair_df[self.tile_pair_df["dir_name"].isin(train_dir)]
            elif split == "valid":
                #ALL data
                if self.standard == "all":
                    self.tile_pair_df = self.all_tile_pair[self.all_tile_pair["dir_name"].isin(valid_dir)]
                elif self.standard == "silver":
                    tile_pairs_silver =  self.tile_pair_df[self.tile_pair_df["dice"]>0.2]
                    self.tile_pair_df = tile_pairs_silver[tile_pairs_silver["dir_name"].isin(valid_dir)]
                elif self.standard == "gold":
                    tile_pairs_gold =  self.tile_pair_df[self.tile_pair_df["dice"]>0.6]
                    self.tile_pair_df = tile_pairs_gold[tile_pairs_gold["dir_name"].isin(valid_dir)]
            elif split == "test":
                #ALL data
                if self.standard == "all":
                    self.tile_pair_df = self.all_tile_pair[self.all_tile_pair["dir_name"].isin(test_dir)]
                elif self.standard == "silver":
                    tile_pairs_silver =  self.tile_pair_df[self.tile_pair_df["dice"]>0.2]
                    self.tile_pair_df = tile_pairs_silver[tile_pairs_silver["dir_name"].isin(test_dir)]
                elif self.standard == "gold":
                    tile_pairs_gold =  self.tile_pair_df[self.tile_pair_df["dice"]>0.6]
                    self.tile_pair_df = tile_pairs_gold[tile_pairs_gold["dir_name"].isin(test_dir)]

        self.all_tiles = [self.all_tile_pair.iloc[i]["pair_name"] for i in range(len(self.all_tile_pair))]
    def __len__(self):
        return len(self.tile_pair_df)

    def __getitem__(self, idx):
        pair = self.tile_pair_df.iloc[idx]


        surrounding_tiles = get_image_roi(pair["pair_name"].split("_")[0], pair["pair_name"].split("_")[1], self.all_tiles, self.dir_path, pair)
        # Load and unpack binary comet data
        
        pkl_data = unpack_and_load(os.path.join(pair["dir_name"], pair["pair_name"] + "_comet_binary_thres_labels.pkl.gz"))
        mask = pkl_data["comet_array_binary"]  
        # mask out non-cell region
        # Map channels to their respective labels             
        cell_masks = [pkl_data["labels_dapi"] if channel in ["DAPI", "TRITIC", "Cy5", "Ki67_1:150 - TRITC"] else pkl_data["labels_dapi_expanded"]
                            for channel in common_channel_list]    
        cell_masks = np.stack(cell_masks, axis=-1)   

        labels_dapi = pkl_data['labels_dapi']
        labels_dapi_expanded = pkl_data['labels_dapi_expanded']                        
            
        if self.mask_noncell:
            mask[cell_masks==0]=0 # Shape: (H, W, C)

        if self.cell_mask_label:
            labeled_nuclei = label(labels_dapi) # Shape: (H, W)
            labeled_nuclei_props = regionprops(labeled_nuclei)

            labeled_cell = label(labels_dapi_expanded) # Shape: (H, W)
            labeled_cell_props = regionprops(labeled_cell)            
            
            mask_new=np.zeros_like(mask) # Shape: (H, W, C)

            # Process each labeled region
            for label_prop_mode, label_props in (("nuclei", labeled_nuclei_props),
                                                 ("cell", labeled_cell_props)):
                for region_idx, region in enumerate(label_props):
                    region_mask = region.convex_image # Shape: (region_height, region_width)                
                    region_bbox= region.bbox #minr, minc, maxr, maxc 
                    region_area_bbox=region.area_bbox

                    mask_bbox = mask[region_bbox[0]:region_bbox[2], region_bbox[1]:region_bbox[3], :] # Shape: (region_height, region_width, C)
                    region_ratio_list=(mask_bbox[region_mask].sum(axis=0)/region_mask.sum()) # Shape: (C,)
                    channel_idx_select=[]
                    # print(list((zip(common_channel_list, region_ratio_list))))
                    for channel_idx, (channel, region_ratio) in enumerate(zip(common_channel_list, region_ratio_list)):
                        valid=False
                        if label_prop_mode=="nuclei":
                            if channel in ["DAPI", "TRITIC", "Cy5", "Ki67_1:150 - TRITC"]:
                                valid=True
                        elif label_prop_mode=="cell":
                            if channel not in ["DAPI", "TRITIC", "Cy5", "Ki67_1:150 - TRITC"]:
                                valid=True
                        else:
                            raise ValueError(label_prop_mode)
                        
                        if valid:
                            if region_ratio>(0.2 if channel in ["Ki67_1:150 - TRITC"] else 0.5):
                                channel_idx_select.append(channel_idx)
                    channel_idx_select=np.array(channel_idx_select)

                    if len(channel_idx_select)>0:

                        # Initialize a full-sized mask with the same shape as the original image
                        full_region_mask = np.zeros((mask.shape[0], mask.shape[1], mask.shape[2]), dtype=bool)  # Shape: (H, W, C)

                        for ch in channel_idx_select:
                            full_region_mask[region_bbox[0]:region_bbox[2], region_bbox[1]:region_bbox[3], ch] = region_mask  # Shape: (H, W, C)                        

                        # Now you can use full_region_mask to index into mask_new directly
                        mask_new[full_region_mask] = 1  # Update the mask_new in the full image coordinates



            mask=mask_new


            augmented = self.transform(image=surrounding_tiles, mask=mask)
            img = augmented['image']
            img = img.astype('float32')
            img = img.transpose(2, 0, 1)  # Convert to CHW format
            mask = augmented['mask']
            mask = mask.astype('float32')
            mask = mask.transpose(2, 0, 1)  # Convert to CHW format
            return img, mask, {'img_id': pair.name}
            
        