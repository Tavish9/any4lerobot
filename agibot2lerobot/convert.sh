export HDF5_USE_FILE_LOCKING=FALSE
export RAY_DEDUP_LOGS=0
python agibot_h5.py \
    --src-path /path/to/AgiBotWorld-Beta/ \
    --output-path /path/to/local \
    --eef-type gripper \
    --cpus-per-task 3