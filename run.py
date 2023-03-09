from src.dataset_formatting import Utils

if __name__ == '__main__': # Needed for any multiprocessing

    Utils.format_forecasting_data('./data/forecasting', debug=False)
    Utils.format_anomaly_data('./data/anomaly_detection', debug=False)
