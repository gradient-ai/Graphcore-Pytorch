from pathlib import Path
import subprocess
import json
import os
import yaml
import logging
from examples_utils.paperspace_utils.dataset_upload_checker import check_files_match_metadata


logger = logging.getLogger()
logger.setLevel(logging.INFO)
# Check that the datasets have mounted as expected

# Gather the datasets expected from the settings.yaml
with open("settings.yaml") as f:
    my_dict = yaml.safe_load(f)
    datasets = my_dict["integrations"].keys()


def check_files_exist(files: [str], dirname: str):
    dirpath = Path(dirname)
    sub_directories = [str(f) for f in dirpath.iterdir() if f.is_dir()]
    for filename in files:
        full_path = str(dirpath/filename)
        if full_path not in sub_directories:
            logging.warning(filename + " not found in " + dirname)
        else:
            dataset_sub_directories = [str(f) for f in Path(full_path).iterdir()]
            if full_path+"/gradient_dataset_metadata.json" in dataset_sub_directories:
                check_files_match_metadata(full_path,False)
            else:
                logging.warning("Metadata file not found in "+ full_path)

# Check that dataset exists and if a metadata file is found check that all files in the metadata file exist
check_files_exist(datasets, "./datasets")

# Check that files are symlinked correctly - this needs to be manually edited for each runtime
expected_exe_cache = ["fine-tuning-bert", "kge_training", "not_here"]
check_files_exist(expected_exe_cache, "/tmp/exe_cache")

#Check that the number of detected IPUs is correct
pod_type = os.getenv("GRAPHCORE_POD_TYPE")
expected_ipu_num = pod_type.replace("pod","")

num_ipus = os.getenv("NUM_AVAILABLE_IPU", "0")

if expected_ipu_num != num_ipus:
    logger.warning("Incorrect number of IPUs found "+ num_ipus+" expected "+ expected_ipu_num)
else:
    logger.info("Correct number IPUs found")