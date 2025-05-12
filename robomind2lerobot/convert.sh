export HDF5_USE_FILE_LOCKING=FALSE
export RAY_DEDUP_LOGS=0
python robomind_h5.py \
    --src-path /path/to/robomind/ \
    --output-path /path/to/local \
    --benchmark benchmark1_1_release \
    --embodiments agilex_3rgb franka_1rgb franka_3rgb franka_fr3_dual tienkung_gello_1rgb tienkung_prod1_gello_1rgb tienkung_xsens_1rgb ur_1rgb \
    --cpus-per-task 2
