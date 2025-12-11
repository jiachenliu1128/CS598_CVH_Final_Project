import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
import warnings
warnings.filterwarnings("ignore")
import argparse
import gc
import multiprocessing as mp
import shutil
import time

import albumentations as A
import cv2
import dicomsdl
import numpy as np
import nvidia.dali as dali
import pandas as pd
import pydicom
import torch
from albumentations.pytorch.transforms import ToTensorV2
from joblib import Parallel, delayed
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from settings import SETTINGS
from timm.data import resolve_data_config
from timm.models import create_model

BATCH_SIZE = 2
THRES = 0.31
AUTO_THRES = False
AUTO_THRES_PERCENTILE = 0.97935

J2K_SUID = '1.2.840.10008.1.2.4.90'
J2K_HEADER = b"\x00\x00\x00\x0C"
JLL_SUID = '1.2.840.10008.1.2.4.70'
JLL_HEADER = b"\xff\xd8\xff\xe0"
SUID2HEADER = {J2K_SUID: J2K_HEADER, JLL_SUID: JLL_HEADER}
VOILUT_FUNCS_MAP = {'LINEAR': 0, 'LINEAR_EXACT': 1, 'SIGMOID': 2}
VOILUT_FUNCS_INV_MAP = {v: k for k, v in VOILUT_FUNCS_MAP.items()}
# roi detection
ROI_YOLOX_INPUT_SIZE = [416, 416]
ROI_YOLOX_CONF_THRES = 0.5
ROI_YOLOX_NMS_THRES = 0.9
ROI_YOLOX_HW = [(52, 52), (26, 26), (13, 13)]
ROI_YOLOX_STRIDES = [8, 16, 32]
ROI_AREA_PCT_THRES = 0.04
# model
MODEL_INPUT_SIZE = [2048, 1024]
N_CPUS = min(4, mp.cpu_count())
N_CHUNKS = 2
RM_DONE_CHUNK = True

#############################################



def make_uid_transfer_dict(df, dcm_root_dir):
    machine_id_to_transfer = {}
    machine_id = df.machine_id.unique()
    for i in machine_id:
        row = df[df.machine_id == i].iloc[0]
        sample_dcm_path = os.path.join(dcm_root_dir, str(row.patient_id),
                                       f'{row.image_id}.dcm')
        dicom = pydicom.dcmread(sample_dcm_path)
        machine_id_to_transfer[i] = dicom.file_meta.TransferSyntaxUID
    return machine_id_to_transfer


class ValTransform:

    def __init__(self):
        self.transform_fn = A.Compose([ToTensorV2(transpose_mask=True)])

    def __call__(self, img):
        return self.transform_fn(image=img)['image']


class RSNADataset(Dataset):

    def __init__(self, df, img_root_dir, transform_fn=None):
        self.img_paths = []
        self.transform_fn = transform_fn
        self.df = df
        for i in tqdm(range(len(df))):
            patient_id = df.at[i, 'patient_id']
            image_id = df.at[i, 'image_id']
            img_name = f'{patient_id}@{image_id}.png'
            img_path = os.path.join(img_root_dir, img_name)
            self.img_paths.append(img_path)
        print(f'Done loading dataset with {len(self.img_paths)} samples.')

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = cv2.imread(img_path)
        if img is None:
            print('ERROR:', img_path)
        if self.transform_fn:
            img = self.transform_fn(img)
        return img

    def get_df(self):
        return self.df


def parse_args():
    parser = argparse.ArgumentParser(
        'Generate and write training bash script.')
    parser.add_argument('--mode',
                        type=str,
                        default='trained',
                        choices=['trained'],
                        help='')
    args = parser.parse_args()
    return args


def main(args):
    TORCH_MODEL_CKPT_PATH = os.path.join(SETTINGS.ASSETS_DIR, 'trained',
                                         'best_convnext_fold_0.pth.tar')

    CSV_PATH = os.path.join(SETTINGS.PROCESSED_DATA_DIR,
                            'classification', 'rsna-breast-cancer-detection', 'cleaned_label.csv')
    CLEANED_IMAGES_DIR = os.path.join(SETTINGS.PROCESSED_DATA_DIR,
                                      'classification',
                                      'rsna-breast-cancer-detection',
                                      'cleaned_images')

    ###########################################################################
    global_df = pd.read_csv(CSV_PATH)
    all_patients = list(global_df.patient_id.unique())
    num_patients = len(all_patients)

    num_patients_per_chunk = num_patients // N_CHUNKS + 1
    chunk_patients = [
        all_patients[num_patients_per_chunk * i:num_patients_per_chunk *
                     (i + 1)] for i in range(N_CHUNKS)
    ]
    print(f'PATIENT CHUNKS: {[len(c) for c in chunk_patients]}')

    pred_dfs = []
    for chunk_idx, chunk_patients in enumerate(chunk_patients):
        df = global_df[global_df.patient_id.isin(chunk_patients)].reset_index(
            drop=True)
        print(
            f'Processing chunk {chunk_idx} with {len(chunk_patients)} patients, {len(df)} images'
        )
        if len(df) == 0:
            continue

        dataset = RSNADataset(df,
                              CLEANED_IMAGES_DIR,
                              transform_fn=ValTransform())
        dataloader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            drop_last=False,
        )

        model_info = {
            'model_name': 'convnext_small.fb_in22k_ft_in1k_384',
            'num_classes': 1,
            'in_chans': 3,
            'global_pool': 'max',
        }
        print(f'Loading model from {TORCH_MODEL_CKPT_PATH}')
        model = create_model(
            model_info['model_name'],
            num_classes=model_info['num_classes'],
            in_chans=model_info['in_chans'],
            pretrained=False,
            checkpoint_path=TORCH_MODEL_CKPT_PATH,
            global_pool=model_info['global_pool'],
        )
        data_config = resolve_data_config({}, model=model)
        print('Data config:', data_config)
        mean = np.array(data_config['mean']) * 255
        std = np.array(data_config['std']) * 255
        print(f'mean={mean}, std={std}')
        mean_tensor = torch.FloatTensor(mean).reshape(1, 3, 1, 1).cuda()
        std_tensor = torch.FloatTensor(std).reshape(1, 3, 1, 1).cuda()
        model.eval()
        model.cuda()

        features_save_dir = os.path.join(SETTINGS.SUBMISSION_DIR, 'features')
        os.makedirs(features_save_dir, exist_ok=True)

        all_probs = []
        with torch.inference_mode():
            batch_idx = 0
            for batch in tqdm(dataloader):
                batch = batch.cuda().float()
                batch = (batch - mean_tensor) / std_tensor
                
                features = model.forward_features(batch)
                pooled_features = model.forward_head(features, pre_logits=True)
                logits = model.head.fc(pooled_features)
                
                probs = logits.sigmoid()[:, 0]
                probs = probs.cpu().numpy()
                all_probs.append(probs)
                
                pooled_features_cpu = pooled_features.cpu()
                for i in range(pooled_features_cpu.shape[0]):
                    img_idx = batch_idx * BATCH_SIZE + i
                    if img_idx < len(df):
                        patient_id = df.at[img_idx, 'patient_id']
                        image_id = df.at[img_idx, 'image_id']
                        feature_filename = f'{patient_id}@{image_id}.pt'
                        feature_path = os.path.join(features_save_dir, feature_filename)
                        torch.save(pooled_features_cpu[i], feature_path)
                
                batch_idx += 1

        all_probs = np.concatenate(all_probs, axis=0)
        all_probs = np.nan_to_num(all_probs, nan=0.0, posinf=None, neginf=None)
        assert all_probs.shape[0] == len(df)

        df['preds'] = all_probs
        pred_dfs.append(df)
        print(f'DONE CHUNK {chunk_idx} with {len(df)} samples')
        print(f'Features saved to: {features_save_dir}')
        del model
        gc.collect()
        torch.cuda.empty_cache()
        print('-----------------------------\n\n')

    pred_df = pd.concat(pred_dfs).reset_index(drop=True)

    # 定义要保留的列（你列出的全部字段 + preds）
    keep_cols = [
        'site_id', 'patient_id', 'image_id', 'laterality', 'view', 'age',
        'cancer', 'biopsy', 'invasive', 'BIRADS', 'implant',
        'density', 'machine_id', 'difficult_negative_case',
        'preds'
    ]

    # 过滤一下，某些列在 test 集可能缺失
    keep_cols = [c for c in keep_cols if c in pred_df.columns]

    image_level_df = pred_df[keep_cols].copy()

    SAVE_SUBMISSION_CSV_PATH = os.path.join(
        SETTINGS.SUBMISSION_DIR,
        'image_level_predictions.csv'
    )
    os.makedirs(SETTINGS.SUBMISSION_DIR, exist_ok=True)

    print("Saving image-level predictions to:", SAVE_SUBMISSION_CSV_PATH)
    image_level_df.to_csv(SAVE_SUBMISSION_CSV_PATH, index=False)


if __name__ == '__main__':
    args = parse_args()
    main(args)