FROM pytorch/pytorch:2.5.0-cuda12.4-cudnn9-devel

# Install git
RUN apt-get update && apt-get install -y git

# Install flash-attn
RUN git clone https://github.com/Dao-AILab/flash-attention.git && \
    cd flash-attention && \
    git checkout c1d146cbd5becd9e33634b1310c2d27a49c7e862 && \
    MAX_JOBS=2 python setup.py install && \
    cd hopper && \
    MAX_JOBS=2 python setup.py install

# Install xformers from GitHub. Use development binary until v0.0.29 comes out
RUN pip install --pre -U xformers

# ...
RUN pip install pytest