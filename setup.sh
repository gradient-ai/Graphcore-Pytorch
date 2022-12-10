#!/bin/bash
# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
# Script to be sourced on launch of the Gradient Notebook


# called from root folder in container
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

symlink-public-resources() {
    public_source_dir=${1}
    target_dir=${2}
    echo "Symlinking - ${public_source_dir} to ${target_dir}"

    # Make sure it exists otherwise you'll copy your current dir
    mkdir -p ${public_source_dir}
    cd ${public_source_dir}
    find -type d -exec mkdir -p "${target_dir}/{}" \;
    #find -type f -not -name "*.lock" -exec cp -sP "${PWD}/{}" "${target_dir}/{}" \;
    find -type f -not -name "*.lock" -print0 | xargs -0 -P 50 -I {} sh -c "cp -sP \"${PWD}/{}\" \"${target_dir}/{}\""
    cd -
}

DETECTED_NUMBER_OF_IPUS="4"

IPU_ARG=${1:-"${DETECTED_NUMBER_OF_IPUS}"}

export NUM_AVAILABLE_IPU=${IPU_ARG}
export GRAPHCORE_POD_TYPE="pod${IPU_ARG}"

export POPLAR_EXECUTABLE_CACHE_DIR="/tmp/exe_cache"
export DATASET_DIR="/tmp/dataset_cache"
export CHECKPOINT_DIR="/tmp/checkpoints"

# mounted public dataset directory (path in the container)
# in the Paperspace environment this would be ="/datasets"
export PUBLIC_DATASET_DIR="/datasets"

# Fine-tuning BERT uses the HF Squad dataset
export HUGGINGFACE_HUB_CACHE="/tmp/huggingface_caches"
export TRANSFORMERS_CACHE="/tmp/huggingface_caches/checkpoints"
export HF_DATASETS_CACHE="/tmp/huggingface_caches/datasets"

export POPTORCH_CACHE_DIR="${POPLAR_EXECUTABLE_CACHE_DIR}"
export POPTORCH_LOG_LEVEL=ERR

prepare_datasets(){
    # symlink exe_cache files
    symlink-public-resources "${PUBLIC_DATASET_DIR}/exe_cache" $POPLAR_EXECUTABLE_CACHE_DIR
    # Symlink squad
    symlink-public-resources "${PUBLIC_DATASET_DIR}/huggingface_caches/datasets/squad" "${HF_DATASETS_CACHE}/squad"
    # symlink local dataset used by vit-model-training notebook
    symlink-public-resources "${PUBLIC_DATASET_DIR}/datasets/chest-xray-nihcc" "${DATASET_DIR}/chest-xray-nihcc"
}

nohup prepare_datasets & tail -f nohup.out &
