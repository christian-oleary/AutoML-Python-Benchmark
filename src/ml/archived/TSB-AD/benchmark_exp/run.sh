#!/bin/bash

MODELS=('TimeSeriesODModel' 'PyCaretADModel' 'LunarADModel')

run_hpo_and_evaluation() {
    local model=$1
    local mode=$2

    if [ "$mode" == "uni" ]; then
        mode_ID="U"
    elif [ "$mode" == "multi" ]; then
        mode_ID="M"
    else
        echo "Invalid mode: ${mode}. Use 'uni' or 'multi'."
        exit 1
    fi

    python HP_Tuning_${mode_ID}.py \
        --dataset_dir "../Datasets/TSB-AD-${mode_ID}/" \
        --file_list "../Datasets/File_List/TSB-AD-${mode_ID}-Tuning.csv" \
        --save_dir "eval/HP_tuning/${mode}/" \
        --AD_Name "${model}"

    python Run_Detector_${mode_ID}.py \
        --dataset_dir "../Datasets/TSB-AD-${mode_ID}/" \
        --file_list "../Datasets/File_List/TSB-AD-${mode_ID}-Eva.csv" \
        --score_dir "eval/score/${mode}/" \
        --save_dir "eval/metrics/${mode}/" \
        --save True \
        --AD_Name "${model}"
}

for model in "${MODELS[@]}"; do
    echo "Running HPO and evaluation for ${model} in univariate mode..."
    run_hpo_and_evaluation "${model}" "uni"
    echo "Running HPO and evaluation for ${model} in multivariate mode..."
    run_hpo_and_evaluation "${model}" "multi"
done
