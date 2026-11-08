# Build with build.sh: it selects a verified, digest-pinned linux/amd64 image.
# No workspace data, checkpoint, Python packages, or GPU driver is installed here.
ARG CUDA_IMAGE
FROM ${CUDA_IMAGE}

ARG MODE
ARG CUDA_RELEASE
ARG CUDA_IMAGE
ARG NCU_RELEASE=2024.1.1
ARG NCU_PACKAGE_VERSION=2024.1.1.4-1
LABEL org.opencontainers.image.description="Vista inference build and profiling tools; GPU execution validation pending" \
      org.opencontainers.image.base.name="${CUDA_IMAGE}" \
      drivebench.mode="${MODE}"

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Exact profiler versions from NVIDIA's signed Ubuntu 22.04 apt repository.
# CUDA 11.7's bundled Compute 2022.2.1 cannot profile Hopper. Put the pinned
# binaries ahead of CUDA's wrapper, which otherwise chooses the latest install.
RUN test "${MODE}" = upstream -o "${MODE}" = upstream-cu118 -o "${MODE}" = modern \
    && test "${CUDA_RELEASE}" = 11.7 -o "${CUDA_RELEASE}" = 11.8 -o "${CUDA_RELEASE}" = 12.4 \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates build-essential git rsync ffmpeg \
        python3.10 python3.10-dev python3.10-venv \
        "nsight-compute-${NCU_RELEASE}=${NCU_PACKAGE_VERSION}" \
        nsight-systems-2024.2.3=2024.2.3.38-242334140272v0 \
    && mkdir -p /opt/drivebench-container \
    && dpkg-query -W -f='${Package}\t${Version}\t${Architecture}\n' \
        > /opt/drivebench-container/packages.tsv \
    && cp /etc/os-release /opt/drivebench-container/os-release \
    && rm -rf /var/lib/apt/lists/*

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/opt/nvidia/nsight-compute/${NCU_RELEASE}:/opt/nvidia/nsight-systems/2024.2.3/target-linux-x64:/usr/local/cuda/bin:${PATH}

# These checks require no GPU. The project bootstrap and preflight run later
# with a real H100 and preserve their own execution/profiler evidence.
RUN python3.10 -c 'import sys; assert sys.version_info[:2] == (3, 10); print(sys.version)' \
        | tee /opt/drivebench-container/python-version.txt \
    && nvcc --version | tee /opt/drivebench-container/nvcc-version.txt \
    && grep -F "release ${CUDA_RELEASE}," /opt/drivebench-container/nvcc-version.txt \
    && ncu --version | tee /opt/drivebench-container/ncu-version.txt \
    && grep -F "${NCU_RELEASE}" /opt/drivebench-container/ncu-version.txt \
    && nsys --version | tee /opt/drivebench-container/nsys-version.txt \
    && grep -F '2024.2.3' /opt/drivebench-container/nsys-version.txt

WORKDIR /workspace
CMD ["bash"]
