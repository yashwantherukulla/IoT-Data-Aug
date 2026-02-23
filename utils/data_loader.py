import json
import pathlib
import re
import shutil
import numpy as np

RAW_DATA_PATH = pathlib.Path(r"C:\Users\eruku\Tech\Everything AI\Projects\IoT Data Aug\data\raw\HAR")

def clean_data(data_path: pathlib.Path):
    """
    For each proband directory under data_path:
    - Delete 'images' and 'videos' subdirectories
    - In the 'data' subdirectory, delete any files/folders not starting with 'acc' or 'gyr'
    """
    for proband_dir in sorted(data_path.iterdir()):
        if not proband_dir.is_dir():
            continue

        # Remove images and videos directories
        for dir_name in ("images", "videos"):
            target = proband_dir / dir_name
            if target.exists():
                shutil.rmtree(target)
                print(f"Deleted: {target}")

        # Keep only acc_* and gyr_* entries in the data directory
        data_dir = proband_dir / "data"
        if data_dir.exists():
            for entry in list(data_dir.iterdir()):
                if not entry.name.startswith(("acc", "gyr")) or "csv" not in entry.name:
                    if entry.is_dir():
                        shutil.rmtree(entry)
                    else:
                        entry.unlink()
                    print(f"Deleted: {entry}")

def reorganize_data(data_path: pathlib.Path):
    """
    For each proband directory under data_path:
    - Move all acc_* and gyr_* files from the 'data' subdirectory to the proband directory
    - Delete the now-empty 'data' subdirectory
    """
    for proband_dir in sorted(data_path.iterdir()):
        if not proband_dir.is_dir():
            continue

        data_dir = proband_dir / "data"
        if data_dir.exists():
            for entry in list(data_dir.iterdir()):
                if entry.name.startswith(("acc", "gyr")):
                    target = proband_dir / entry.name
                    shutil.move(str(entry), str(target))
                    print(f"Moved: {entry} to {target}")

            # Remove the empty data directory
            if not any(data_dir.iterdir()):
                data_dir.rmdir()
                print(f"Deleted empty directory: {data_dir}")

def split_data(raw_data_path: pathlib.Path, num_ppl_train: int):
    """
    Split the proband directories into training and validation sets.
    - Randomly select num_ppl_train probands for the training set, and the rest for the validation set.
    - Move the selected proband directories into 'train' and 'val' subdirectories under raw_data_path.
    """
    proband_dirs = [d for d in sorted(raw_data_path.iterdir()) if d.is_dir()]
    np.random.shuffle(proband_dirs)

    train_dirs = proband_dirs[:num_ppl_train]
    val_dirs = proband_dirs[num_ppl_train:]

    train_path = raw_data_path / "train"
    val_path = raw_data_path / "val"
    train_path.mkdir(exist_ok=True)
    val_path.mkdir(exist_ok=True)

    for dir in train_dirs:
        shutil.move(str(dir), str(train_path / dir.name))
        print(f"Moved to train: {dir}")

    for dir in val_dirs:
        shutil.move(str(dir), str(val_path / dir.name))
        print(f"Moved to val: {dir}")

def unzip_data(raw_data_path: pathlib.Path):
    """
    Unzip all .zip files in the raw_data_path and delete the original .zip files.
    """
    for zip_file in raw_data_path.glob("proband*/*.zip"):
        extract_dir = zip_file.with_suffix("")  # same name, no .zip
        extract_dir.mkdir(exist_ok=True)

        shutil.unpack_archive(
            str(zip_file),
            str(extract_dir)
        )

        zip_file.unlink()
        print(f"Unzipped {zip_file.name} into {extract_dir} and deleted the zip")


def filter_activities(raw_data_path: pathlib.Path):
    """
    For each proband directory under raw_data_path:
    - Keep only acc_{xyz}_* and gyr_{xyz}_* folders where xyz is one of:
      climbingup, climbingdown, jumping, running, walking
    - Delete all other acc_* and gyr_* folders.
    """
    allowed_activities = {"climbingup", "climbingdown", "jumping", "running", "walking"}

    for proband_dir in sorted(raw_data_path.iterdir()):
        if not proband_dir.is_dir():
            continue

        for entry in sorted(proband_dir.iterdir()):
            if not entry.is_dir() or not entry.name.startswith(("acc_", "gyr_")):
                continue

            # entry.name is like acc_climbingup_csv or gyr_running_csv
            parts = entry.name.split("_")
            # parts[0] = "acc"/"gyr", parts[1] = activity, parts[2+] = rest
            activity = parts[1] if len(parts) >= 2 else ""

            if activity not in allowed_activities:
                shutil.rmtree(entry)
                print(f"Deleted: {entry}")


def filter_upperarm(raw_data_path: pathlib.Path):
    """
    For each proband directory under raw_data_path:
    - In each acc_{xyz}_csv folder, keep only:
        - CSV files whose name ends with '_upperarm.csv'
        - README files (any file whose name starts with 'readme', case-insensitive)
    - Delete all other files.
    """
    for proband_dir in sorted(raw_data_path.iterdir()):
        if not proband_dir.is_dir():
            continue

        for acc_dir in sorted(proband_dir.iterdir()):
            if not acc_dir.is_dir() or not acc_dir.name.startswith("acc_"):
                continue

            for entry in list(acc_dir.iterdir()):
                if entry.is_dir():
                    continue

                is_upperarm_csv = entry.suffix == ".csv" and entry.stem.endswith("_upperarm")
                is_readme = entry.name.lower().startswith("readme")

                if not is_upperarm_csv and not is_readme:
                    entry.unlink()
                    print(f"Deleted: {entry}")
                elif is_readme and entry.name != "readMe.txt":
                    new_path = entry.parent / "readMe.txt"
                    entry.rename(new_path)
                    print(f"Renamed: {entry} -> {new_path}")


def extract_freq_info(raw_data_path: pathlib.Path):
    """
    For each proband directory under raw_data_path:
    - In each acc_{xyz}_csv folder, read readMe.txt
    - Find the frequency listed under the acc_{xyz}_upperarm.csv entry
    - Write an info.json file in that folder with the field "freq" (float, in Hz)
    """
    for proband_dir in sorted(raw_data_path.iterdir()):
        if not proband_dir.is_dir():
            continue

        for acc_dir in sorted(proband_dir.iterdir()):
            if not acc_dir.is_dir() or not acc_dir.name.startswith("acc_"):
                continue

            readme_path = acc_dir / "readMe.txt"
            if not readme_path.exists():
                print(f"No readMe.txt found in {acc_dir}, skipping.")
                continue

            content = readme_path.read_text(encoding="utf-8")

            # Match the block starting at the _upperarm.csv filename entry
            # and capture the frequency on the following lines
            match = re.search(
                r"acc_[^_]+_upperarm\.csv\s*\n(?:.*\n)*?>\s*frequency:\s*([\d.]+)\s*Hz",
                content
            )

            if match:
                freq = float(match.group(1))
                info = {"freq": freq}
                info_path = acc_dir / "info.json"
                info_path.write_text(json.dumps(info, indent=4), encoding="utf-8")
                print(f"Written: {info_path} (freq={freq} Hz)")
                readme_path.unlink()
                print(f"Deleted: {readme_path}")
            else:
                print(f"Could not find upperarm frequency in {readme_path}")


if __name__ == "__main__":
    clean_data(RAW_DATA_PATH)
    reorganize_data(RAW_DATA_PATH)
    unzip_data(RAW_DATA_PATH)
    filter_activities(RAW_DATA_PATH)
    filter_upperarm(RAW_DATA_PATH)
    extract_freq_info(RAW_DATA_PATH)
    # split_data(RAW_DATA_PATH, num_ppl_train=10)
