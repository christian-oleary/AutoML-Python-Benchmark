"""Main module for Git repository analysis."""

import sys
from pathlib import Path

from ml.logs import logger
from ml.sca.analysis import Analysis


def run_analysis():
    """Run static code analysis on a Git repository."""
    # Check number of inputs
    if len(sys.argv) == 0 or len(sys.argv) > 4:
        logger.error('Usage: python -m ml.sca [repository_path] [output_dir_path]')
        sys.exit(1)

    # Validate input_dir
    input_dir = sys.argv[1]
    if not Path(input_dir).is_dir():
        logger.error(f'Path must be a directory: {input_dir}')
        sys.exit(1)

    # Validate output_dir
    if len(sys.argv) == 3:
        output_dir = sys.argv[2]
        try:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        except FileNotFoundError:
            logger.error(f'Invalid output directory: {output_dir}')
            sys.exit(1)
    else:
        output_dir = Path('results/sca/')

    # Run analysis on the specified directory
    logger.info(f'Running static code analysis on {input_dir}')
    _ = Analysis(input_dir, output_dir=output_dir).run()
    logger.info(f'Saved results to {output_dir}')


if __name__ == "__main__":
    run_analysis()
