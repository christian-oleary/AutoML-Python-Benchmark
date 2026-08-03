# -*- coding: utf-8 -*-
# Author: Qinghua Liu <liu.11085@osu.edu>
# License: Apache-2.0 License

import argparse
import itertools
import os
import random
import time
from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
import torch

from TSB_AD.evaluation.metrics import get_metrics
from TSB_AD.HP_list import Multi_algo_HP_dict
from TSB_AD.model_wrapper import (
    run_Semisupervise_AD,
    run_Unsupervise_AD,
    Semisupervise_AD_Pool,
    Unsupervise_AD_Pool,
)
from TSB_AD.utils.slidingWindows import find_length_rank

# seeding
seed = 2024
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

print("CUDA available: ", torch.cuda.is_available())
print("cuDNN version: ", torch.backends.cudnn.version())

if __name__ == '__main__':

    Start_T = time.time()
    ## ArgumentParser
    parser = argparse.ArgumentParser(description='HP Tuning')
    parser.add_argument('--dataset_dir', type=str, default='../Datasets/TSB-AD-M/')
    parser.add_argument(
        '--file_list', type=str, default='../Datasets/File_List/TSB-AD-M-Tuning.csv'
    )
    parser.add_argument('--save_dir', type=str, default='eval/HP_tuning/multi/')
    parser.add_argument('--AD_Name', type=str, default='IForest')
    args = parser.parse_args()

    file_list = pd.read_csv(args.file_list)['file_name'].values

    Det_HP = Multi_algo_HP_dict[args.AD_Name]

    keys, values = zip(*Det_HP.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    write_csv = []

    # Find existing results
    df_existing = pd.DataFrame()
    csv_path = Path(f'{args.save_dir}/{args.AD_Name}.csv')
    if csv_path.exists():
        df_existing = pd.read_csv(csv_path)

    # Add results to list and update after each iteration
    all_results_list: list = []
    if len(df_existing) > 0:
        for _, row in df_existing.iterrows():
            all_results_list.append(row.tolist())

    # Iterate over files
    for i, filename in enumerate(file_list):
        logger.info(f'Processing {i+1}/{len(file_list)}: {filename} by {args.AD_Name}')

        file_path = os.path.join(args.dataset_dir, filename)
        df = pd.read_csv(file_path).dropna()
        data = df.iloc[:, 0:-1].values.astype(float)
        label = df['Label'].astype(int).to_numpy()

        slidingWindow = find_length_rank(data[:, 0].reshape(-1, 1), rank=1)
        train_index = filename.split('.')[0].split('_')[-3]
        data_train = data[: int(train_index), :]

        # Iterate over HP combinations
        for index, params in enumerate(combinations):
            # Check if result already exists
            if len(df_existing) > 0:
                mask = (df_existing['file'] == filename) & (df_existing['HP'] == str(params))
                if mask.any():
                    logger.trace(f'Skipping {filename} with HP {params} as it already exists')
                    continue

            # Otherwise, run the model
            logger.debug(f'params ({index + 1}/{len(combinations)}): {params}')
            if args.AD_Name in Semisupervise_AD_Pool:
                output = run_Semisupervise_AD(args.AD_Name, data_train, data, **params)
            elif args.AD_Name in Unsupervise_AD_Pool:
                output = run_Unsupervise_AD(args.AD_Name, data, **params)
            else:
                raise Exception(f"{args.AD_Name} is not defined")

            try:
                evaluation_result = get_metrics(output, label, slidingWindow=slidingWindow)
                logger.debug(f'evaluation_result: {evaluation_result}')
                scores = list(evaluation_result.values())
            except Exception:
                scores = [0] * 9

            scores.insert(0, params)  # type: ignore
            scores.insert(0, filename)
            write_csv.append(scores)

            # Temp Save
            cols = list(evaluation_result.keys())
            cols.insert(0, 'HP')
            cols.insert(0, 'file')
            df_results = pd.DataFrame(write_csv, columns=cols)

            Path(args.save_dir).mkdir(parents=True, exist_ok=True)
            df_results.to_csv(f'{args.save_dir}/{args.AD_Name}.csv', index=False)
