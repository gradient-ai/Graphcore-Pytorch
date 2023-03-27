#!/bin/bash

set -x

symlink-public-resources() {
    public_source_dir=${1}
    target_dir=${2}

    local -i COUNTER=0
    # need to wait until the dataset has been mounted (async on Paperspace's end)
    while [ $COUNTER -lt 300 ] && ( [ ! -d ${public_source_dir} ] || [ -z "$(ls -A ${public_source_dir})" ] )
    do
        echo "Waiting for dataset "${public_source_dir}" to be mounted..."
        sleep 1
        ((COUNTER++))
    done

    if [ $COUNTER -eq 300 ]; then
        echo "Warning! Abandoning symlink - source Dataset ${public_source_dir} has not been mounted & populated after 5m."
        return
    fi

    echo "Symlinking - ${public_source_dir} to ${target_dir}"

    # Make sure it exists otherwise you'll copy your current dir
    mkdir -p ${target_dir}
    workdir="/fusedoverlay/workdirs/${public_source_dir}"
    upperdir="/fusedoverlay/upperdir/${public_source_dir}"
    mkdir -p ${workdir}
    mkdir -p ${upperdir}
    fuse-overlayfs -o lowerdir=${public_source_dir},upperdir=${upperdir},workdir=${workdir} ${target_dir}

}

if [ ! "$(command -v fuse-overlayfs)" ]
then
    echo "fuse-overlayfs not found installing - please update to our latest image"
    apt update -y
    apt install -o DPkg::Lock::Timeout=120 -y psmisc libfuse3-dev fuse-overlayfs
fi

# RRR install lmpe and dependencies, needed for gptj
sudo apt-get update && sudo apt-get install libopenmpi-dev
sudo apt-get install liblmpe-dev libnl-3-dev


echo "Starting preparation of datasets"
# symlink exe_cache files
exe_cache_source_dir="${PUBLIC_DATASETS_DIR}/poplar-executables-pytorch-3-2"
symlink-public-resources "${exe_cache_source_dir}" $POPLAR_EXECUTABLE_CACHE_DIR

# Symlink squad
symlink-public-resources "${PUBLIC_DATASETS_DIR}/squad" "${HF_DATASETS_CACHE}/squad"
symlink-public-resources "${PUBLIC_DATASETS_DIR}/glue" "${HF_DATASETS_CACHE}/glue"

# symlink local dataset used by vit-model-training notebook
symlink-public-resources "${PUBLIC_DATASETS_DIR}/chest-xray-nihcc-3" "${DATASETS_DIR}/chest-xray-nihcc-3"
# Symlink OGB Wiki dataset and checkpoint
symlink-public-resources "${PUBLIC_DATASETS_DIR}/ogbl_wikikg2_custom" "${DATASETS_DIR}/ogbl_wikikg2_custom"

# pre-install the correct version of optimum for this release
python -m pip install "optimum-graphcore>=0.5, <0.6"

echo "Finished running setup.sh."
# Run automated test if specified
if [[ "$1" == "test" ]]; then
    bash /notebooks/.gradient/automated-test.sh "${@:2}"
elif [[ "$2" == "test" ]]; then
    bash /notebooks/.gradient/automated-test.sh "${@:3}"
fi
