# Grasp poses generation for robotic hand
In this project, it is proposed to generate various poses for grasping objects.

As a robotic hand, we use a three-fingered gripper (in the picture below):

<img src="https://github.com/BE2R-Lab-RND-AI-Grasping/DexGraspNet/blob/docs/setup-instructions/Images/Hand_original.png" alt="The three-fingered gripper" width="300">

To accelerate the generation, we will replace some of the hand parts with primitives such as capsules:

<img src="https://github.com/BE2R-Lab-RND-AI-Grasping/DexGraspNet/blob/docs/setup-instructions/Images/Hand_primitives.png" alt="The primitive gripper" width="300">

## INSTALLATION
### Preparation
We use Docker. For correct working with GPU on Docker you need install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). It is installed in the host.

### Docker Container Creation
So let's start by creating a container.
1) Clone this repo to your system
```bash
git clone https://github.com/BE2R-Lab-RND-AI-Grasping/DexGraspNet.git
```
3) Go to the root folder of the repository and build a Docker container using the following command:
```bash
docker build --pull --rm -f 'Dockerfile' -t 'dexgraspnet:latest' '.'
```
3) To run this code in VS Code, use the `Reopen in Container` function.
4) When starting the container for the first time, you need to install TorchSDF (TorchSDF is a custom version of Kaolin)
```bash
cd thirdparty
git clone https://github.com/wrc042/TorchSDF.git
cd TorchSDF
bash install.sh
```
### Installing dependencies
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
> `Pytorch Kinematics` was already installed during creating container.

# GRASP GENERATION
This code generates different optimized hand poses for objects and saves these values to an .npy file (a separate file for each object). These files are saved in the `data/graspdata` folder.

> Full dataset and object meshes you can find [HERE](https://mirrors.pku.edu.cn/dl-release/DexGraspNet-ICRA2023/). Files from `dexgraspnet.tar.gz` put to the folder `data/dataset`, Files from `meshdata.tar.gz` put to the folder `data/meshdata`

You should have the container running now. Run file:
```bash
cd grasp_generation/
export CUDA_VISIBLE_DEVICES=0
python scripts/generate_grasps.py --all
```
> We have one GPU, so `CUDA_VISIBLE_DEVICES=0`, if you have more GPUs write it in this form `export CUDA_VISIBLE_DEVICES=x,x,x` (instead `x` use your GPUs ID).

> The generation process takes a lot of time. It's fine if the progress bar shows 0%. For faster generation, you can leave several objects in the 'data/meshdata` folder.

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

## Result visualization
To visualize the obtained grasping poses, we will transfer the generated position data of all links to the original hand XML and render it in Mujoco.

<img src="https://github.com/BE2R-Lab-RND-AI-Grasping/DexGraspNet/blob/docs/setup-instructions/Images/hand_and_obj_07_05_2025.png" alt="The three-fingered gripper" width="300">

In this image, the hand is not perfectly gripping the hammer. It is necessary to specify the contact points on the hand more precisely. **Work is currently underway on this.**

A sequence of generated roses:

<img src="https://github.com/BE2R-Lab-RND-AI-Grasping/DexGraspNet/blob/docs/setup-instructions/Images/Hand_grasp_poses_gif.gif" alt="The three-fingered gripper" width="300">

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
