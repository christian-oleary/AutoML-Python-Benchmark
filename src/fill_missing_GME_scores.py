import os
import numpy as np
import pandas as pd
from scipy.stats import gmean

def mae_over(actual, predicted):
        """Overestimated predictions (from Grimes et al. 2014)"""
        errors = predicted - actual
        positive_errors = np.clip(errors, 0, errors.max())
        return np.mean(positive_errors)

def mae_under(actual, predicted):
    """Underestimated predictions (from Grimes et al. 2014)"""
    errors = predicted - actual
    negative_errors = np.clip(errors, errors.min(), 0)
    return np.absolute(np.mean(negative_errors))

def calculate_missing_gme_scores(results_dir, data_file):
    df = pd.read_csv(data_file)

    if 'ISEM_prices' in data_file:
        test_df = pd.read_csv('X_test.csv')
        test_df = test_df.drop('applicable_date', axis=1)
        actual = test_df.values.flatten()
    else:
        test_df = df.tail(int(len(df)* 0.2))
        raise NotImplementedError()

    for dirpath, _, filenames in os.walk(results_dir):

        result_files = [
            f for f in filenames
            if f.endswith('.csv') and f.split('.')[0] in [
                'autogluon', 'autokeras', 'autots', 'evalml', 'fedot', 'flaml', 'ludwig', 'pycaret', 'test'
            ]
        ]

        if len(result_files) > 0:
            for filename in result_files:
                print('filename', filename)
                path = os.path.join(dirpath, filename)
                df = pd.read_csv(path)
                preds = pd.read_csv(os.path.join(dirpath, 'predictions.csv'), header=None).values.flatten()

                # Add missing metrics
                df['MAEunder'] = mae_over(actual, preds)
                df['MAEover'] = mae_under(actual, preds)

                if 'GME' in df.columns:
                    df = df.drop('GME', axis=1)

                # Correct calculations
                df['GM-MAE-SR'] = df.apply(lambda row: gmean([row['MAE'], 1 - row['Spearman Correlation']]), axis=1)
                df['GM-MASE-SR'] = df.apply(lambda row: gmean([row['MASE'], 1 - row['Spearman Correlation']]), axis=1)

                df.to_csv(path, index=False)


if __name__ == '__main__':
    for results_dir, data_file in [
        ('./results/univariate_forecasting/ISEM_prices_2020_all_year/', './data/univariate_electricity/ISEM_prices_2020_all_year.csv'),
    ]:
        if os.path.exists(results_dir):
            calculate_missing_gme_scores(results_dir, data_file)
