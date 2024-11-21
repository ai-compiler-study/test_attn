#!/bin/bash

if [ -z "${SETUP_SCRIPT}" ]; then
  echo "ERROR: SETUP_SCRIPT is not set"
  exit 1
fi

. "${SETUP_SCRIPT}"

test_attn_dir=$(dirname "$(readlink -f "$0")")/../..
cd ${test_attn_dir}

# Install third-party libraries
# (Required) FBGEMM
# (Required) kernels
# (Required) generative-recommenders
# (Optional) ThunderKittens
# (Optional) cutlass-kernels
# (Optional) flash-attention
python install.py --all
