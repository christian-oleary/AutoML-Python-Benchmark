"""Main module for Git repository analysis."""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from ml.logs import logger
from ml.sca.analysis import Analysis
from ml.sca.repo import prepare_repositories


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

    # Prepare repositories
    if not os.getenv('GITHUB_TOKEN'):
        load_dotenv()

    prepare_repositories(
        clone_path=Path(input_dir),
        pull_changes=False,
        results_dir=Path(output_dir, 'git_analysis'),
        skip_existing=True,
        github_token=os.getenv('GITHUB_TOKEN'),
    )

    # Run analysis on the specified directory
    logger.info(f'Running static code analysis on {input_dir}')
    _ = Analysis(input_dir, output_dir=output_dir).run()
    logger.success(f'Saved results to "{output_dir}"')


if __name__ == "__main__":
    run_analysis()
