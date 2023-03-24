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

# Check that dataset exists and
dirname = Path("/datasets")
datasets_sub_directories = [str(f) for f in dirname.iterdir() if f.is_dir()]
print(datasets_sub_directories)
for dataset in datasets:
    full_path = str(dirname/dataset)
    if full_path not in datasets_sub_directories:
        print(dataset + " not found")

# Using script from examples-utils check that the metadata files are correct
# Do not need to run full hash checks

# Check that files are symlinked correctly
expected_files = ["tmp/exe_cache/fine-tuning-bert"]
dirname = Path("/tmp")
tmp_sub_directories = [str(f) for f in dirname.iterdir() if f.is_dir()]
print(tmp_sub_directories)
for file_name in expected_files:
    if file_name not in tmp_sub_directories:
        print(file_name + " not mounted")

#Check that the number of detected IPUs is correct
pod_type = os.getenv("GRAPHCORE_POD_TYPE")
expected_ipu_num = pod_type.replace("pod","")

num_ipus = os.getenv("NUM_AVAILABLE_IPU")

if expected_ipu_num != num_ipus:
    print("Incorrect number of IPUs found "+ num_ipus+" expected "+ expected_ipu_num)
else:
    print("Correct number IPUs found")