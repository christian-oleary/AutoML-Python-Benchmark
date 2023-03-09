import logging
import sys

from src.dataset_formatting import Utils

# Set up logging
logger = logging.getLogger('Benchmark')
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel(level=logging.DEBUG)

if __name__ == '__main__': # Needed for any multiprocessing

    # Download data if needed
    gather_metadata = False # If true, will parse all .tsf files and output a .csv file of dataset descriptors
    Utils.format_forecasting_data('./data/forecasting', gather_metadata=gather_metadata, debug=False)
    Utils.format_anomaly_data('./data/anomaly_detection', debug=False)
