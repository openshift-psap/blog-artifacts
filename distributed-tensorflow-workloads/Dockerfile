FROM registry.redhat.io/ubi8/ubi:latest
MAINTAINER Courtney Pacheco <cpacheco@redhat.com>

# Set CUDA version to use
ENV CUDA_MAJOR_VERSION=10
ENV CUDA_MAJOR_RELEASE=1

# For TensorFlow, set the versions of required packages, as well as the arch and Python version
ENV BAZEL_VERSION=3.1.0
ENV NUMPY_VERSION=1.18.5
ENV TF_VERSION=2.3.0
ENV ARCH=x86_64
ENV PYTHON_VERSION=36

# Copy the CUDA repo file
COPY cuda.repo /etc/yum.repos.d/cuda.repo

# Copy cuDNN
ARG CUDNN_TARBALL=cudnn-${CUDA_MAJOR_VERSION}.${CUDA_MAJOR_RELEASE}-linux-x64-v8*.tgz
COPY ${CUDNN_TARBALL} /tmp

# Copy NCCL
ARG NCCL_TARBALL=nccl*${CUDA_MAJOR_VERSION}.${CUDA_MAJOR_RELEASE}*.txz
COPY ${NCCL_TARBALL} /usr/local

# Install EPEL, python3-devel, python3-pip, gcc, git, patch, and wget
RUN dnf -y install https://dl.fedoraproject.org/pub/epel/epel-release-latest-8.noarch.rpm && \
    dnf -y install gcc \
                   git \
                   patch \
                   python3-devel \
                   python3-pip \
                   unzip \
                   wget && \
    dnf clean all && \
    rm -rf /var/cache/dnf*

# Install CUDA and TensorFlow related packages
RUN dnf -y install cuda-libraries-${CUDA_MAJOR_VERSION}-${CUDA_MAJOR_RELEASE} \
                   cuda-${CUDA_MAJOR_VERSION}-${CUDA_MAJOR_RELEASE} \
                   dkms \
                   nvidia-kmod && \
    dnf clean all && \
    rm -rf /var/cache/dnf*

# Install cuDNN
RUN cd /tmp && \
    tar xvf cudnn-${CUDA_MAJOR_VERSION}.${CUDA_MAJOR_RELEASE}-linux-x64-v8*.tgz && \
    cp cuda/lib64/* /usr/local/cuda/lib64 && \
    cp cuda/include/* /usr/local/cuda/include && \
    rm -rf cuda && \
    rm -rf cudnn-${CUDA_MAJOR_VERSION}.${CUDA_MAJOR_RELEASE}-linux-x64-v8*.tgz

# Install NCCL
RUN cd /usr/local && \
    tar xvf nccl*.txz && \
    rm -rf nccl*.txz && \
    mv nccl* nccl

# Install TensorFlow 2.x requried packages with pip3
RUN pip3 install numpy==${NUMPY_VERSION} && \
    pip3 install absl-py astor gast keras keras_preprocessing protobuf wheel

# Install Bazel
RUN cd /tmp && \
    wget https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh && \
    sh bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh && \
    rm -rf bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh

# Create symlinks for /usr/bin/python and /usr/lib/python3.6
RUN ln -s /usr/bin/python3 /usr/bin/python && \
    ln -s /usr/lib/python3.6m /usr/lib/python3.6

# Configure the TensorFlow 2.x build
ENV GCC_HOST_COMPILER_PATH=/usr/bin/gcc
ENV CC_OPT_FLAGS='-march=native'
ENV TF_DOWNLOAD_CLANG=0
ENV TF_ENABLE_XLA=1
ENV TF_NEED_COMPUTECPP=0
ENV TF_NEED_CUDA=1
ENV TF_NEED_MPI=0
ENV TF_NEED_OPENCL=0
ENV TF_NEED_OPENCL_SYCL=0
ENV TF_NEED_ROCM=0
ENV TF_NEED_TENSORRT=0
ENV TF_SET_ANDROID_WORKSPACE=0
ENV PYTHON_BIN_PATH=/usr/bin/python
ENV PYTHON_LIB_PATH=/usr/lib/python3.6

#For CUDA 10.1, cublas_api.h is actually in the CUDA 10.2 directory
ENV TF_CUDA_PATHS='/usr/local/cuda,/usr/include,/usr/lib64,/usr/local/cuda-10.2/targets/x86_64-linux/include,/usr/local/cuda-10.2/targets/x86_64-linux/lib,/usr/local/nccl'
ENV TF_CUDA_VERSION=10.1
ENV TF_CUDNN_VERSION=8
ENV TF_CUDA_COMPUTE_CAPABILITIES='3.5,7.0'
ENV TF_CUDA_CLANG=0
ENV TF_NCCL_VERSION=''

# Clone git repo for TensorFlow and configure the build
RUN cd /opt && \
    git clone https://github.com/tensorflow/tensorflow.git && \
    cd tensorflow && \
    git checkout tags/v${TF_VERSION} && \
    ./configure

# Build the TensorFlow pip3 package
RUN cd /opt/tensorflow && \
    bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package && \
    ./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg

# Install the TensorFlow pip3 package
RUN pip3 install /tmp/tensorflow_pkg/tensorflow-${TF_VERSION}-cp${PYTHON_VERSION}-cp${PYTHON_VERSION}m-linux_${ARCH}.whl

# Acquire benchmarks and install benchmark packages
RUN pip3 install tf-slim tensorflow_model_optimization && \
    pip3 install tensorflow-datasets==3.0.0 && \
    pip3 install gin-config==0.1.1 && \
    cd /opt && \
    git clone https://github.com/tensorflow/models

ENV PYTHONPATH=/opt/models
WORKDIR /opt/models/official/vision/image_classification

# Set environment vars
ENV MODEL_DIR=/opt/models
ENV DATA_DIR=/opt/models/official/vision/image_classification

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV NVIDIA_REQUIRE_CUDA "cuda>=10.1 brand=tesla,driver>=418,driver<419 brand=tesla,driver>=440,driver<441"
