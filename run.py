import logging
import sys

from src.utils import Utils

# Set up logging
logger = logging.getLogger('Benchmark')
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel(level=logging.DEBUG)

# Download data if needed
Utils.format_forecasting_data('./data/forecasting', debug=False)
Utils.format_anomaly_data('./data/anomaly_detection', debug=False)
