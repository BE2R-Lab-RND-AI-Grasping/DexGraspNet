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

# GRASP GENERATION
This code generates different optimized hand poses for objects and saves these values to an .npy file (a separate file for each object). These files are saved in the `data/graspdata` folder.

You should have the container running now. Run file:
```bash
cd grasp_generation/
export CUDA_VISIBLE_DEVICES=0
python scripts/generate_grasps.py --all
```
> We have one GPU, so `CUDA_VISIBLE_DEVICES=0`, if you have more GPUs write it in this form `export CUDA_VISIBLE_DEVICES=x,x,x` (instead `x` use your GPUs ID).
This generation takes a lot of time, It's ok if you see 0% in you progress bar. For faster generation, you can leave several objects in the 'data/meshdata` folder.

> Adjust parameters `batch_size_each` to get the desired amount of data. Turn down `max_total_batch_size` if CUDA runs out of memory. Remember to change the random seed `seed` to get different results. Other numeric parameters are magical and we don't recommend tuning them.

## Data results
Each file like `core-bottle-1a7ba1f4c892e2da30711cdbdbc73924.npy` contains a list of data dicts. Each dict represents one synthesized grasp:
* scale: The scale of the object.
* qpos: The final grasp pose g=(T,R,θ), which is logged as a dict:
 * WRJTx,WRJTy,WRJTz: Translations in meters.
 * WRJRx,WRJRy,WRJRz: Rotations in euler angles, following the xyz convention.
 * robot0:XXJn: Articulations passed to the forward kinematics system.
* qpos_st: The initial grasp pose logged like qpos. This entry will be removed after grasp validation.
* energy,E_fc,E_dis,E_pen,E_spen,E_joints: Final energy terms. These entries will be removed after grasp validation.

## Error Solving
If the `divide by zero` error appears, replace the 'v = (d1[is_ab] / (d1[is_ab] - d3[is_ab])).reshape((-1, 1))' in the `triangles.py`:
```bash
denom = d1[is_ab] - d3[is_ab]
denom = np.where(np.abs(denom) < 1e-8, 1e-8, denom)  # Replacing values that are too small
v = (d1[is_ab] / denom).reshape((-1, 1))
```

# QUICK EXAMPLE
In this example, you do not need to follow the steps described above (create a container, etc.)
There are some objects, a hand (shadowhand), and pre-generated hand poses for this object are just loaded. The grasp pose is visualized by the `Trimesh` library.

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

> For the full DexGraspNet dataset, go to our [project page](https://pku-epic.github.io/DexGraspNet/) for download links. Decompress dowloaded packages and link (or move) them to corresponding path in `data`.
