"""Utility functions."""

import os
import shutil
from pathlib import Path
from time import sleep

from loguru import logger


class Utils:
    """Utility functions."""

    @staticmethod
    def delete_paths(*paths: str | Path, wait_seconds: float = 0.5):
        """Delete paths representing files and/or folders.

        >>> os.makedirs('test_folder', exist_ok=True)
        >>> Utils.delete_paths('test_folder')
        >>> Path('test_folder').exists()
        False

        :param str | Path | None path: File or folder to delete
        :param float wait_seconds: Wait time before retrying, defaults to 0.5
        """
        for path in [p for p in paths if p is not None]:
            logger.debug(f'Deleting path: "{path}"')
            # Attempt to delete file/folder
            try:
                Utils._attempt_delete_path(path)
            except PermissionError:
                sleep(wait_seconds)  # Wait for file locks to release in seconds
                Utils._attempt_delete_path(path)

            # Raise error if checkpoint still exists
            if os.path.exists(path):
                raise FileExistsError(f'Failed to delete {path}')

    @staticmethod
    def _attempt_delete_path(path: str | Path):
        """Attempt to delete file or folder.

        >>> os.makedirs('test_folder', exist_ok=True)
        >>> Utils._attempt_delete_path('test_folder')
        >>> Path('test_folder').exists()
        False

        :param str | Path path: File or folder to delete
        """
        # Remove folder
        if os.path.isdir(path):
            # Recursively delete all files
            folders: list = []
            for root, dirs, files in os.walk(path):
                for file in files:
                    os.remove(os.path.join(root, file))
                for folder in dirs:
                    folders.append(os.path.join(root, folder))
            # Delete folders
            folders.append(path)
            for folder in folders:
                try:
                    shutil.rmtree(folder, ignore_errors=False)
                except FileNotFoundError:
                    pass
                # except PermissionError:
                #     os.chmod(folder, 0o777)
                #     shutil.rmtree(folder, ignore_errors=False)
        # Remove file
        elif os.path.isfile(path):
            os.remove(path)
        # Ensure path does not exist
        if Path(path).exists():
            raise PermissionError(f'Failed to delete {path}')

    @staticmethod
    def find_files_by_extension(
        input_dir: str | Path, extension: str, unique_files: bool = True, parent_id: bool = True
    ) -> list[Path]:
        """Find files in a directory with a specific extension.

        :param str | Path input_dir: Path to directory
        :param bool unique_files: Return only uniquely named files, defaults to True
        :param bool parent_id: Include parent directory name in file ID, defaults to True
        :return list: List of paths to .wav files
        """
        paths = []
        seen_ids = set()
        # Find all files with the specified extension
        for file_path in Path(input_dir).rglob(f'*.{extension}'):
            if unique_files:
                # Get unique ID for each file
                file_id = str(file_path.name)
                if parent_id:
                    file_id = str(file_path.parent.stem) + file_id
                # Skip if already seen
                if file_id in seen_ids:
                    continue
                seen_ids.add(file_id)

            paths.append(file_path)
        return paths
