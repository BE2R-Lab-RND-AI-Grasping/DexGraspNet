# Implementation of the three-fingered model

## In xml file:

Rename names of mesh files according to mesh names in xml file.

Replace fingers bodies in xml-file with capsules. The string must be in the following format:

`<geom name="primitive" type="capsule" size="0.015 0.02" pos="0 0 0.04"/>`

Change the body's format to 

`<body name="" pos="" quat="">`

## In `generate_grasps.py` or `main.py`:

Replace `joint_names` with your xml file joint names. In function `generate` replace `mjcf_path` and `mesh_path` according to your files location.

## In `initialisations.py`:

Replace `joint_angles_mu` with your initial joint positions.

## In `hand_model.py`:

First, replace `joint_angles` with your initial joint positions replace string 
`link_mesh = tm.load_mesh(os.path.join(mesh_path, visual.geom_param[0].split(":")[1]+".obj"), process=False)` 
with 
`link_mesh = tm.load_mesh(os.path.join(mesh_path, visual.geom_param[0]+".stl"), process=False)`.

Second, in function `__init__` replace list in string `if link_name in []:` with names of your bodies which is being use in `cal_distance` function and is not phalanges. In function `cal_distance` replace list in `if link_name in []:` with your bodies wich schouldn't calculate distance. 

## In `contact_points.json`: 

In this file write contact points in all bodies as a dictionary where keys are names of bodies and vaule is empty list (if no points in body) or list with contact points coordinates (for phalanges bodies). 

## In `penetration_points.json`:

In this file write penetration points in all bodies as a dictionary where keys are names of bodies and vaule is empty list (if no points in body) or list with penetration points coordinates (for phalanges bodies). 

## In `visualize_reult.py` and `visualize_hand_model.py`:

Replace `joint_name` and `hand_model` according your model. In `visualize_hand_model.py` you can radius of keypoints for better visualization.
