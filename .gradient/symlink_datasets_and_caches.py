import json
import time
from pathlib import Path
import subprocess


# read in symlink config file
with open(f"{Path(__file__).parent.absolute().as_posix()}/symlink_config.json", "r") as f:
    config = json.loads(f.read())

# loop through each key-value pair
# the key is the target directory, the value is a list of source directories
for target_dir, source_dirs_list in config.items():
    # need to wait until the dataset has been mounted (async on Paperspace's end)
    source_dirs_exist_paths = []
    for source_dir in source_dirs_list:
        source_dir_path = Path(source_dir)
        COUNTER = 0
        # 300s/5m timeout for waiting for the dataset
        keep_waiting = ( (COUNTER < 300) 
                          # wait for the dataset to exist and be populated/non-empty
                          and not (source_dir_path.exits() and any(source_dir_path.iterdir())) )
        while keep_waiting: 
            print(f"Waiting for dataset {source_dir_path.as_posix()} to be mounted...")
            time.sleep(1)
            COUNTER += 1

        # dataset doesn't exist after 300s, skip it
        if COUNTER == 300:
            print(f"Abandoning symlink! - source dataset {source_dir} has not been mounted & populated after 5 minutes.")
            break
        else:
            print(f"Found dataset {source_dir}")
            source_dirs_exist_paths.append(source_dir)
    
    # create overlays for source dataset dirs 
    if len(source_dirs_exist_paths) > 0:
        print(f"Symlinking - {source_dirs_exist_paths} to {target_dir}")
        print("-" * 100)

        Path(target_dir).mkdir(parents=True, exist_ok=True)

        workdir_path = Path("/fusedoverlay/workdirs" + target_dir)
        workdir_path.mkdir(parents=True, exist_ok=True)
        upperdir_path = Path("/fusedoverlay/upperdir" + target_dir) 
        upperdir_path.mkdir(parents=True, exist_ok=True)

        lowerdirs = ":".join(source_dirs_exist_paths)
        overlay_command = f"fuse-overlayfs -o lowerdir={lowerdirs},upperdir={upperdir_path.as_posix()},workdir={workdir_path.as_posix()} {target_dir}"
        subprocess.run(overlay_command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

