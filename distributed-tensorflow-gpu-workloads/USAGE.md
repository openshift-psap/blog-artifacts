# Operator Installation Playbook

This brief guide explains how to use the `operator_installation` playbook to install/uninstall the three operators necessary to run distributed TensorFlow workloads on the GPU: Node Feature Discovery (NFD), Special Resource Operator (SRO), and Kubeflow (via Open Data Hub).

Note that you can define which operator(s) you want to install or uninstall, so if you already have one or more operators installed to the cluster, you can choose which ones to install or uninstall.

## User Mode

When running this playbook, users are required to input the user mode, which is either `install` or `uninstall`. This user mode is passed in via a JSON file or the `--extra-vars` option on the command line.

### Installation

#### Requirements

Before you begin an install of one or more operators, make sure you know which version of OpenShift you are running. (e.g., 4.3, 4.4, etc.). Note that these playbooks only support OpenShift 4.x!

If you are using AWS, the playbook will run checks on your nodes to see if you have at least one GPU-enabled instance available. If no GPU nodes are available and you are installing more than just NFD and Kubeflow, the install will fail.

#### If None of the Operators are Already Deployed to your Cluster

If you don't have any of the three operators already installed to your cluster, you can simply run

```bash
$ ansible-playbook -i hosts play.yaml --extra-vars="{ mode: 'install', oc_release: '4.3' }"
```

...except replace the `oc_release` variable with the major release of OpenShift you're using. This is important!

If you are not using OpenShift on AWS, please pass in the `use_aws` extra var and set it equal to `no`. e.g.,

```bash
$ ansible-playbook -i hosts play.yaml --extra-vars="{ mode: 'install', oc_release: '4.3', use_aws: 'no' }"
```

This extra var will disable AWS-specific checks for GPU-enabled instances.

#### If you Already have Some Operators Installed

If you have one or more operators already installed to your cluster, you can set which operators to install. There are three extra vars you can use:

  - `install_nfd` - Installs NFD only
  - `install_sro` - Installs SRO and NFD, if NFD is not already installed (because SRO requires NFD)
  - `install_kubeflow` - Installs Kubeflow only


For example, if you already have NFD installed to your cluster, you can run:

```bash
$ ansible-playbook -i hosts play.yaml --extra-vars="{ mode: 'install', oc_release: '4.3', install_nfd: 'no', install_sro: 'yes', install_kubeflow: 'yes' }"
```

You can also set `use_aws` equal to `no` in this situation, too.

#### Force Reinstall

If you want to force a reinstall of one or more operators (i.e., remove the operator and reinstall it), use the `force_reinstall` extra var. You can combine the `force_reinstall` with the three extra vars described above (`install_nfd`, `install_sro`, and `install_kubeflow`). Again, you can set AWS checks on or off via `use_aws`.

### Uninstalling

To uninstall one or more operators, you will need to set the user mode to `uninstall`. But much like you can control the installation of which operators you want to install, you can also control which operators to *uninstall*. Such extra vars are:

  - `uninstall_nfd` - Uninstalls NFD and SRO (because SRO depends on NFD)
  - `uninstall_sro` - Uninstalls SRO only
  - `uninstall_kubeflow` - Uninstalls Kubeflow only

Example command:

```bash
$ ansible-playbook -i hosts play.yaml --extra-vars="{ mode: 'uninstall', oc_release: '4.3', uninstall_nfd: 'no', uninstall_sro: 'yes', uninstall_kubeflow: 'yes' }"
```

Note that uninstalling NFD automatically uninstalls SRO because SRO requires NFD in order to function.
