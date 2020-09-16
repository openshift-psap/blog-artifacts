# Distributed TensorFlow Workloads for OpenShift using the GPU

### Requirements

You must have already downloaded the NCCL and cuDNN tarballs to this directory. NCCL can be downloaded from NVIDIA's site after signing in and visiting [this page](https://developer.nvidia.com/nccl). Similarly, you can download cuDNN from NVIDIA's site after logging in. You can find cuDNN downloads [on this page](https://developer.nvidia.com/cudnn).

You will also need to log into `registry.redhat.io` with your credentials in order to pull the ubi8 image.

Finally, you will need to have a `cuda.repo` file that can be pulled into the image. Your file will look something like this:

```
[CUDA]
name=CUDA
baseurl=<url-to-cuda-rhel8-repo>
enabled=1
gpgcheck=0
```

GPG checking is optional and you will be able to find more information about the `.repo` file [on this page](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&target_distro=RHEL&target_version=8&target_type=rpmnetwork).


### Testing the Image you Built Outside of OpenShift

If you wish to test the image you built before using it in OpenShift, you will first need to ensure you have an NVIDIA driver installed on your machine. (CUDA packages are not required, however.) Then, you will need to add your NVIDIA devices to the command line and run with the `--privileged` flag. For example, if you have a single GPU,

```bash
$ podman run --privileged -it --device /dev/nvidia0:/dev/nvidia0 dc58ce12ff9dad /bin/bash
```

NVIDIA devices will typically be labled as `/dev/nvidia0`, `/dev/nvidia1`, `/dev/nvidia2`, etc..

Try running `nvidia-smi` in the image to see if your GPUs show up.

### Notes

Note that everything here has been tested for TensorFlow 2.3.0 on ubi8 as of this blog post. Inevitably, TensorFlow will be updated and you may wish to use another version.

TensorFlow 2.3.0 officially supports CUDA 10.1, which is why CUDA 10.1 is used here. Eventually, TensorFlow maintainers will support CUDA 10.2 and beyond. However, it is worthwhile to note that some CUDA 10.2 packages are actually installed by default, even when you specify to install CUDA 10.1. This is not problematic, though, because these CUDA 10.2 packages simply contain files such as cuBLAS related headers that are not different from their equivalents in CUDA 10.1. Nonetheless, the CUDA shared object libraries are still 10.1 libraries.
