# Distributed TensorFlow Workloads for OpenShift using the GPU

### Requirements

#### Pulling from registry.redhat.io 

Before you begin anything, you will need to log into `registry.redhat.io` with your credentials in order to pull the ubi8 image. To login with podman,

```bash
$ podman login
```

or Docker:

```bash
$ docker login
```

#### Creating a cuda.repo File

Next, you will need to have a `cuda.repo` file that can be pulled into the image during build time. A sample repo template is provided under the `Dockerfiles` directory with the name `cuda.repo.template`. You will be able to find more information about the `.repo` file [on this page](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&target_distro=RHEL&target_version=8&target_type=rpmnetwork).

\*\*Note I left GPG checking as *optional*. You can find GPG key info in the url above.

#### Creating an nvidia-ml.repo file

In addition to the `cuda.repo` file, you will need a repo file to pull in NVIDIA machine learning (ML) packages. To do so, you will need to create an nvidia-ml.repo file. As with the `cuda.repo` template, a template exists for the `nvidia-ml.repo` file. This repo file will allow the image build to pull in the [cuDNN](https://developer.nvidia.com/CUDnn) and [NCCL](https://developer.nvidia.com/nccl) packages, both of which are required to run the distributed machine learning workload.


### Building the Images

You will need to build two images to run the distributed Fashion MNIST benchmark. Note that the first image builds the TensorFlow GPU image itself, while the second image pulls in that TensorFlow image. This process is done to allow users to build a different version of TensorFlow 2.x and use that image as the base image, rather than having to update the file yourself.

To build the TensorFlow GPU image, first ensure you're in the main directory of this repo, then run

```bash
$ export YOUR_TF_IMAGE_TAG="some-repo/some-tag-name:latest"
$ podman build -t ${YOUR_TF_IMAGE_TAG} -f Dockerfiles/Dockerfile.ubi8-tf .
$ podman push ${YOUR_TF_IMAGE_TAG}
```

After that, run:

```bash
$ export YOUR_FASHION_MNIST_TAG="some/repo/tf-fashion-mnist:latest"
$ podman build -t ${YOUR_FASHION_MNIST_TAG} -f Dockerfiles/Dockerfile.fashion-mnist
$ podman push ${YOUR_FASHION_MNIST_TAG}
``` 

If your image repository (or the final image itself) requires a pull secret, make sure to download that pull secret now.

### Create a New OCP Project called 'distributed-tf'

While your image is uploading, you will want to keep all of your OCP work in one namespace, so create the `distributed-tf` namespace that will be referenced by the `TFJob` benchmark.

```bash
$ oc new-project distributed-tf
```

### Ensure you have at Least Two GPU Instances

Also while your image is uploading, you will need at least two GPU instances to run this distributed benchmark. If you're running OCP in AWS, GCP, Azure, etc., create those GPU instances now. (For AWS, `g4dn.*` instances are recommended.) Ensure the nodes are up and running before continuing to the next step.

### Launch the Necessary Operators

Before you create your TFJob, you will need to install the following operators:

  - NFD (Node Feature Discovery) Operator
  - NVIDIA GPU Operator
  - Kubeflow/OpenDataHub Operator

### Add your Newly-Pushed Image Name to the TFJob YAML and Optionally Add your Pull Secret as an OCP 'Secret'

The next step is to add your newly-pushed image name to the TFJob YAML file that will be used to launch the distributed GPU workload job. Additionally, if your image repository requires a pull secret for OCP to pull your final image, create that pull secret now. Call it whatever you like. Make sure to edit the `yaml/fashion-mnist-tfjob.yaml` file with your image name (and, optionally, pull secret name):

```bash
$ # Image adding
$ sed -i "s/YOUR-IMAGE-NAME/${YOUR_FASHION_MNIST_TAG}/g" yaml/fashion-mnist-tfjob.yaml

$ # Optional steps:
$ export MY_OCP_PULL_SECRET_NAME="my-pull-secret"
$ sed -i "s/YOUR-PULL-SECRET/${MY_OCP_PULL_SECRET_NAME}/g" yaml/fashion-mnist-tfjob.yaml
```

Make sure that you set the number of `replicas` (i.e., worker nodes) that you want to use for your benchmark. Currently, the number of replicas is set to `2`. Change to your desired number of GPU instances to use, and make sure to edit the Node Selector at the end of the yaml file. (Note that the last number of the `command` field equals the number of desired workers to use, so ensure that both the parameter and replica numbers match.)


### Ready to Go!

Now create the TFJob and watch the magic begin!

```bash
$ oc create -f yaml/fashion-mnist-tfjob.yaml
```

### Notes

Note that everything here has been tested for TensorFlow 2.3.0 on ubi8 as of this blog post. Inevitably, TensorFlow will be updated and you may wish to use another version.

TensorFlow 2.3.0 officially supports CUDA 10.1, which is why CUDA 10.1 is used here. Eventually, TensorFlow maintainers will support CUDA 10.2 and beyond. However, it is worthwhile to note that some CUDA 10.2 packages are actually installed by default, even when you specify to install CUDA 10.1. This is not problematic, though, because these CUDA 10.2 packages simply contain files such as cuBLAS related headers that are not different from their equivalents in CUDA 10.1. Nonetheless, the CUDA shared object libraries are still 10.1 libraries.
