from pathlib import Path
import subprocess
import json
import os
import yaml

# Check that the datasets have mounted as expected

# Gather the datasets expected from the settings.yaml
with open('settings.yaml') as f:
    my_dict = yaml.safe_load(f)
    datasets = my_dict["integrations"].keys()

def check_files_exist(files: [str], dirname: str):
    dirpath = Path(dirname)
    sub_directories = [str(f) for f in dirpath.iterdir() if f.is_dir()]
    print(sub_directories)
    for filename in files:
        full_path = str(dirpath/filename)
        if full_path not in sub_directories:
            print(filename + " not found")

# Check that dataset exists and
check_files_exist(datasets, "/datasets")

# Using script from examples-utils check that the metadata files are correct
# Do not need to run full hash checks

# Check that files are symlinked correctly
expected_exe_cache = ["fine-tuning-bert", "kge_training"]
check_files_exist(expected_files, "/tmp/exe_cache")

#Check that the number of detected IPUs is correct
pod_type = os.getenv("GRAPHCORE_POD_TYPE")
expected_ipu_num = pod_type.replace("pod","")

num_ipus = os.getenv("NUM_AVAILABLE_IPU")

if expected_ipu_num != num_ipus:
    print("Incorrect number of IPUs found "+ num_ipus+" expected "+ expected_ipu_num)
else:
    print("Correct number IPUs found")