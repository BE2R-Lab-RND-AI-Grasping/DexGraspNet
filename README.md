# DexGraspNet: A Large-Scale Robotic Dexterous Grasp Dataset for General Objects Based on Simulation

## INSTALLATION
### Preparation
We use Docker, so install some needed libs.
1) [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) is installed in the host.


### Container Creation
So let's start by creating a container.
1) Clone this repo
2) Creating a container from a Dockerfile. Located in the root (where the Dockefile is) use
```bash
docker build --pull --rm -f 'Dockerfile' -t 'dexgraspnet' '.'
```
3) For the VS Code - click the `Reopen in container`.
4) When starting the container for the first time, you need to install TorchSDF
```bash
cd thirdparty/TorchSDF
bash install.sh
```

## Quick Example
In this example, an object, a hand, and pre-generated hand poses for this object are  Just loaded. The grasp pose is visualized by the `Trimesh` library.

```bash
conda create -n your_env python=3.7
conda activate your_env

# for quick example, cpu version is OK.
conda install pytorch cpuonly -c pytorch
conda install ipykernel
conda install transforms3d
conda install trimesh
pip install pyyaml
pip install lxml

cd thirdparty/pytorch_kinematics
pip install -e .
```

Then you can run `grasp_generation/quick_example.ipynb`.

For the full DexGraspNet dataset, go to our [project page](https://pku-epic.github.io/DexGraspNet/) for download links. Decompress dowloaded packages and link (or move) them to corresponding path in `data`.
