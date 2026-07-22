# Minimal NVIDIA CUDA runtime on Ubuntu (glibc-based)
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

# Install Python 3.10 and pip (ubuntu 22.04 default is 3.10)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Make 'python' and 'pip' available without the '3.10' suffix
RUN ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

WORKDIR /workspace

# Keep container interactive by default
CMD ["/bin/bash"]