# Change this value from "podman" to "docker" if using Docker still
IMAGE_BUILDER="podman"

# Change this info when building your images
REPOSITORY="MY-REPOSITORY"
ORG="MY-ORG"
TF_IMAGE_NAME="distributed-tensorflow-gpu"
COCO_IMAGE_NAME="tf-coco-util"
IMAGE_TAG="latest"

# Change this info if creating an EC2 volume
AWS_AVAILABILITY_ZONE="us-east-1a"

# No need to modify these values
FULL_TF_IMAGE_NAME=$(REPOSITORY)/$(ORG)/$(TF_IMAGE_NAME):$(IMAGE_TAG)
FULL_COCO_IMAGE_NAME=$(REPOSITORY)/$(ORG)/$(COCO_IMAGE_NAME):$(IMAGE_TAG)
DOCKERFILES_PATH="Dockerfiles"
YAMLS_PATH="yaml"
COCO_LOADER_YAML="$(YAMLS_PATH)/coco/coco_dataset_loading_job.yaml"
TF_RECORD_BUILDER_YAML="$(YAMLS_PATH)/tf_record_builder_job.yaml"

# Main
all:
	build-tf-image

# Image related builds
build-coco-image:
	$(IMAGE_BUILDER) build -t $(FULL_COCO_IMAGE_NAME) -f $(DOCKERFILES_PATH)/Dockerfiles.coco .

build-tf-image:
	$(IMAGE_BUILDER) build -t $(FULL_TF_IMAGE_NAME) -f $(DOCKERFILES_PATH)/Dockerfile.ubi8 .

delete-coco-image:
	$(IMAGE_BUILDER) rmi $(FULL_COCO_IMAGE_NAME) -f

delete-tf-image:
	$(IMAGE_BUILDER) rmi $(FULL_TF_IMAGE_NAME) -f

push-coco-image:
	$(IMAGE_BUILDER) push $(FULL_COCO_IMAGE_NAME)

push-tf-image:
	$(IMAGE_BUILDER) push $(FULL_TF_IMAGE_NAME)

# Data loading and data storage related builds
create-aws-ebs:
	aws ec2 create-volume --volume-type gp2 --size 500 --availability-zone $(AVAILABILITY_ZONE)

create-coco-loader:
	sed -i "s|YOUR-IMAGE|$(FULL_COCO_IMAGE_NAME)|g" $(COCO_LOADER_YAML)
	sed -i "s|YOUR-PULL-SECRET|$(PULL_SECRET_NAME)|g" $(COCO_LOADER_YAML)
	oc create -f $(COCO_LOADER_YAML)

create-tf-record-builder:
	sed -i "s|YOUR-IMAGE|$(FULL_COCO_IMAGE_NAME=)|g" $(TF_RECORD_BUILDER_YAML)
	sed -i "s|YOUR-PULL-SECRET|$(PULL_SECRET_NAME)|g" $(TF_RECORD_BUILDER_YAML)
	oc create -f $(TF_RECORD_BUILDER_YAML)

delete-coco-loader:
	oc delete -f $(COCO_LOADER_YAML)

delete-tf-record-builder:
	oc delete -f $(TF_RECORD_BUILDER_YAML)

create-verification-pod:
	oc create -f $(YAMLS_PATH)/verification/tf_and_coco_verification_test_pod.yaml
 
delete-verification-pod:
	oc delete -f $(YAMLS_PATH)/verification/tf_and_coco_verification_test_pod.yaml
