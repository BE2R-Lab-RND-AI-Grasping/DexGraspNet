# DexGraspNet: A Large-Scale Robotic Dexterous Grasp Dataset for General Objects Based on Simulation

## INSTALLATION
### Preparation
We use Docker, so install some needed libs.
1) For correct working with gpu on Docker you need install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). It is installed in the host.

### Docker Container Creation
So let's start by creating a container.
1) Clone this repo to your system
2) Build Docker container from a Dockerfile. You must be in the root (where the Dockefile is) and use
```bash
docker build --pull --rm -f 'Dockerfile' -t 'dexgraspnet:latest' '.'
```
3) For the VS Code - click the `Reopen in container`.
4) When starting the container for the first time, you need to install TorchSDF (TorchSDF is a custom version of Kaolin)
```bash
cd thirdparty
git clone https://github.com/wrc042/TorchSDF.git
cd TorchSDF
bash install.sh
```
### Common Packages
```bash
conda install pytorch3d

conda install transforms3d
conda install trimesh
conda install plotly

pip install urdf_parser_py
pip install scipy

pip install networkx  # soft dependency for trimesh
conda install rtree  # soft dependency for trimesh
```
`Pytorch Kinematics` was already installed during creating container.

### GRASP GENERATION



## QUICK EXAMPLE
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
