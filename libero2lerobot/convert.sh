export SVT_LOG=1
export HF_DATASETS_DISABLE_PROGRESS_BARS=TRUE
export HDF5_USE_FILE_LOCKING=FALSE

python libero_h5.py \
    --src-paths /path/to/libero/ \
    --output-path /path/to/local \
    --executor local \
    --tasks-per-job 3 \
    --workers 10
