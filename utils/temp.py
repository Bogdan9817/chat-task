from pathlib import Path
import os


def temp_folder_exists(path: str = "temp"):
    def dec(func):
        def wrapper(*args, **kwargs):
            temp_folder = Path(path)
            if not temp_folder.exists or not temp_folder.is_dir():
                os.mkdir(path)
            result = func(*args, **kwargs)
            return result

        return wrapper

    return dec
