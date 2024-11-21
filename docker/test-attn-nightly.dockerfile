ARG BASE_IMAGE=xzhao9/gcp-a100-runner-dind:latest

FROM ${BASE_IMAGE}


ENV CONDA_ENV=pytorch
ENV CONDA_ENV_TRITON_MAIN=triton-main
ENV SETUP_SCRIPT=/workspace/setup_instance.sh
ARG TEST_ATTN_BRANCH=${TEST_ATTN_BRANCH:-main}
ARG FORCE_DATE=${FORCE_DATE}

# Checkout TestAttn and submodules
RUN git clone --recurse-submodules -b "${TEST_ATTN_BRANCH}" --single-branch \
    https://github.com/ai-compiler-study/test_attn.git /workspace/test_attn

# Setup conda env and CUDA
RUN cd /workspace/test_attn && \
    . ${SETUP_SCRIPT} && \
    python tools/python_utils.py --create-conda-env ${CONDA_ENV} && \
    echo "if [ -z \${CONDA_ENV} ]; then export CONDA_ENV=${CONDA_ENV}; fi" >> /workspace/setup_instance.sh && \
    echo "conda activate \${CONDA_ENV}" >> /workspace/setup_instance.sh

RUN cd /workspace/test_attn && \
    . ${SETUP_SCRIPT} && \
    sudo python tools/cuda_utils.py --setup-cuda-softlink

# Install PyTorch nightly and verify the date is correct
RUN cd /workspace/test_attn && \
    . ${SETUP_SCRIPT} && \
    python tools/cuda_utils.py --install-torch-deps && \
    python tools/cuda_utils.py --install-torch-nightly

# Check the installed version of nightly if needed
RUN cd /workspace/test_attn && \
    . ${SETUP_SCRIPT} && \
    if [ "${FORCE_DATE}" = "skip_check" ]; then \
        echo "torch version check skipped"; \
    elif [ -z "${FORCE_DATE}" ]; then \
        FORCE_DATE=$(date '+%Y%m%d') \
        python tools/cuda_utils.py --check-torch-nightly-version --force-date "${FORCE_DATE}"; \
    else \
        python tools/cuda_utils.py --check-torch-nightly-version --force-date "${FORCE_DATE}"; \
    fi

# test_attn library build and test require libcuda.so.1
# which is from NVIDIA driver
RUN sudo apt update && sudo apt-get install -y libnvidia-compute-550 patchelf

# Install test_attn
RUN cd /workspace/test_attn && \
    bash .ci/test_attn/install.sh


# Test test_attn
RUN cd /workspace/test_attn && \
    bash .ci/test_attn/test-install.sh

# Remove NVIDIA driver library - they are supposed to be mapped at runtime
RUN sudo apt-get purge -y libnvidia-compute-550

# Clone the pytorch env as triton-main env, then compile triton main from source
RUN cd /workspace/test_attn && \
    BASE_CONDA_ENV=${CONDA_ENV} CONDA_ENV=${CONDA_ENV_TRITON_MAIN} bash .ci/test_attn/install-triton-main.sh
