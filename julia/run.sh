#!/bin/bash
# First and only argument is model ID
PROJECT_DIR="/home/sdeshmukh/Documents/graphCRNs/julia"
RES_DIR="${PROJECT_DIR}/res"
# List files in RES_DIR/model_id

# For each file, move the current file to the wrk directory
# (create if it doesn't exist), call JCRN with the wrk directory as the arg
# and then move the file back
base_dir=${RES_DIR}/$1
for file in $(find $base_dir -maxdepth 1 -name "*.csv")
do
  new_dir="$(dirname $file)/wrk"
  base_name=$(basename $file)
  # Move file to wrk directory
  echo "Moving $file to $new_dir"
  mkdir -p $new_dir
  mv $file $new_dir
  # Call JCRN with wrk dir as argument
  echo "Running JCRN in $new_dir"
  julia --sysimage ~/julia_sysimages/sys_catalyst.so run_jcrn.jl $1
  # Move file back to original directory
  echo "Moving $new_dir/$base_name back to $base_dir"
  mv $new_dir/$base_name $base_dir
done
