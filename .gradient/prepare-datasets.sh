#!/bin/bash

symlink-public-resources() {
    public_source_dir=${1}
    target_dir=${2}

    # need to wait until the dataset has been mounted (async on Paperspace's end)
    #while [ ! -d "${PUBLIC_DATASET_DIR}/exe_cache" ]
    while [ ! -d ${public_source_dir} ]
    do
        echo "Waiting for dataset "${public_source_dir}" to be mounted..."
        sleep 1
    done

    echo "Symlinking - ${public_source_dir} to ${target_dir}"

    # Make sure it exists otherwise you'll copy your current dir
    mkdir -p ${public_source_dir}
    cd ${public_source_dir}
    find -type d -exec mkdir -p "${target_dir}/{}" \;
    #find -type f -not -name "*.lock" -exec cp -sP "${PWD}/{}" "${target_dir}/{}" \;
    find -type f -not -name "*.lock" -print0 | xargs -0 -P 50 -I {} sh -c "cp -sP \"${PWD}/{}\" \"${target_dir}/{}\""
    cd -
}

echo "Starting preparation of datasets"
# symlink exe_cache files
exe_cache_source_dir="${PUBLIC_DATASET_DIR}/exe_cache_pytorch"
symlink-public-resources "${exe_cache_source_dir}" $POPLAR_EXECUTABLE_CACHE_DIR
# Symlink squad
symlink-public-resources "${PUBLIC_DATASET_DIR}/huggingface_caches/datasets/squad" "${HF_DATASETS_CACHE}/squad"
# symlink local dataset used by vit-model-training notebook
# symlink-public-resources "${PUBLIC_DATASET_DIR}/datasets/chest-xray-nihcc" "${DATASET_DIR}/chest-xray-nihcc"

# pre-install the correct version of optimum for this release
python -m pip install "optimum-graphcore>0.4, <0.5"

echo "Finished running setup.sh."
# Run automated test if specified
if [[ "$1" == "test" ]]; then
    #source .gradient/automated-test.sh "${@:2}"
    bash /notebooks/.gradient/automated-test.sh $2 $3 $4 $5 $6 $7 $8
elif [[ "$2" == "test" ]]; then
    #source .gradient/automated-test.sh "${@:2}"
    bash /notebooks/.gradient/automated-test.sh $3 $4 $5 $6 $7 $8 $9
fi
