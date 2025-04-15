# DexGraspNet: A Large-Scale Robotic Dexterous Grasp Dataset for General Objects Based on Simulation

## INSTALLATION

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
