"""Main module for Git repository analysis."""

import sys
from pathlib import Path

from ml.logs import logger
from ml.sca.analysis import Analysis


def run_analysis():
    """Run static code analysis on a Git repository."""
    # Check if the path is provided
    if len(sys.argv) != 2:
        logger.error('Usage: python -m ml.sca [path]')
        sys.exit(1)
    input_path = sys.argv[1]

    # Check if the path is a directory
    if not Path(input_path).is_dir():
        logger.error(f'Path must be a directory: {input_path}')
        sys.exit(1)

    # Run analysis on the specified directory
    logger.info(f'Running static code analysis on {input_path}')
    results_dir = Path('results', 'sca')
    _ = Analysis(input_path, output_dir=results_dir).run()
    logger.info(f'Saved results to {results_dir}')


if __name__ == "__main__":
    run_analysis()
