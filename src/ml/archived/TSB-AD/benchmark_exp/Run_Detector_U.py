# -*- coding: utf-8 -*-
# Author: Qinghua Liu <liu.11085@osu.edu>
# License: Apache-2.0 License

import argparse
import os
import random
import time

from loguru import logger
import pandas as pd
import numpy as np
import torch

from TSB_AD.evaluation.metrics import get_metrics
from TSB_AD.HP_list import Optimal_Uni_algo_HP_dict
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

    # ArgumentParser
    parser = argparse.ArgumentParser(description='Generating Anomaly Score')
    parser.add_argument('--dataset_dir', type=str, default='../Datasets/TSB-AD-U/')
    parser.add_argument('--file_list', type=str, default='../Datasets/File_List/TSB-AD-U-Eva.csv')
    parser.add_argument('--score_dir', type=str, default='eval/score/uni/')
    parser.add_argument('--save_dir', type=str, default='eval/metrics/uni/')
    parser.add_argument('--AD_Name', type=str, default='IForest')
    args = parser.parse_args()

    target_dir = os.path.join(args.score_dir, args.AD_Name)
    os.makedirs(target_dir, exist_ok=True)

    file_list = pd.read_csv(args.file_list)['file_name'].values
    Optimal_Det_HP = Optimal_Uni_algo_HP_dict[args.AD_Name]
    logger.info(f'Optimal_Det_HP: {Optimal_Det_HP}')

    # Specify path to save results
    csv_path = os.path.join(args.save_dir, f'{args.AD_Name}.csv')
    os.makedirs(args.save_dir, exist_ok=True)

    # Load any existing results to avoid redundant processing
    df_existing = pd.DataFrame()
    if os.path.exists(csv_path):
        logger.debug(f"Existing results found at {csv_path}. Loading existing results.")
        df_existing = pd.read_csv(csv_path)
        all_scores_list = df_existing.values.tolist()

    all_scores_list = []
    for filename in file_list:
        predictions_path = f'{target_dir}/{filename.split(".")[0]}.npy'

        # Check if scores for this file already exist in dataframe
        if len(df_existing) > 0:
            existing_row = df_existing[df_existing['file'] == filename]
            if not existing_row.empty:
                all_scores_list.append(existing_row.values.tolist()[0])  # Record existing scores
                logger.debug(f"Existing scores found for {filename}. Skipping.")
                continue

        # if 'MITDB' in filename: continue

        # Load data
        logger.info(f'Loading: {filename} by {args.AD_Name}')
        file_path = os.path.join(args.dataset_dir, filename)
        df = pd.read_csv(file_path).dropna()
        data = df.iloc[:, 0:-1].values.astype(float)
        label = df['Label'].astype(int).to_numpy()

        # Specify sliding window size and training data
        slidingWindow = find_length_rank(data[:, 0].reshape(-1, 1), rank=1)
        train_index = filename.split('.')[0].split('_')[-3]
        data_train = data[: int(train_index), :]

        # Load predictions and runtime if they already exist
        if os.path.exists(predictions_path):
            print('Loading results...')
            output = np.load(predictions_path)
            with open(predictions_path.replace('.npy', '.txt'), 'r', encoding='utf-8') as time_file:
                run_time = float(time_file.read().strip())
        else:
            # Run the specified anomaly detection algorithm and measure time
            start_time = time.time()
            if args.AD_Name in Semisupervise_AD_Pool:
                output = run_Semisupervise_AD(args.AD_Name, data_train, data, **Optimal_Det_HP)
            elif args.AD_Name in Unsupervise_AD_Pool:
                output = run_Unsupervise_AD(args.AD_Name, data, **Optimal_Det_HP)
            else:
                logger.error(f"{args.AD_Name} is not defined")
                raise Exception(f"{args.AD_Name} is not defined")
            end_time = time.time()
            run_time = end_time - start_time

            # Save predictions and log results
            if isinstance(output, np.ndarray):
                logger.info(
                    f'Success at {filename} using {args.AD_Name} | '
                    f'Time: {run_time:.3f}s at length {len(label)}'
                )
                np.save(predictions_path, output)
                with open(
                    predictions_path.replace('.npy', '.txt'), 'w', encoding='utf-8'
                ) as time_file:
                    time_file.write(str(run_time))
            else:
                logger.error(f'At {filename}: {output}')

        # Evaluate results
        try:
            print('Evaluating results...')
            evaluation_result = get_metrics(output, label, slidingWindow=int(slidingWindow))
            logger.info(f'evaluation_result: {evaluation_result}')
            scores = list(evaluation_result.values())
        except Exception:
            scores = [0] * 9

        scores.insert(0, str(Optimal_Det_HP))  # type: ignore
        scores.insert(0, run_time)  # type: ignore
        scores.insert(0, filename)
        all_scores_list.append(scores)

        # Save results after each iteration
        cols = list(evaluation_result.keys())
        cols.insert(0, 'HP')
        cols.insert(0, 'Time')
        cols.insert(0, 'file')
        df_scores = pd.DataFrame(all_scores_list, columns=cols)
        df_scores.to_csv(csv_path, index=False)
