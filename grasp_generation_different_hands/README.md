# Preparing a new hand to poses generating

## FILES

1) Correct `.XML` file of hand. Hand's meshes must be `.STL`.

    - The coordinates must be correctly positioned: the X-axis along the link, the Z-axis along the axis of rotation of the joint.

    - Give names to all `<body>`, `<joint>`, `<mesh>` and `.STL` files with normal understandable names.

    - `<joint>` must contain the parameters `name` and `range` (range of angles)(even if the range is defined in `<default>`)

    - Set the angles in radians and the path to meshes in this way: 
    ``` bash 
    <compiler angle="radian" meshdir="./mjcf/assets/robotiq/"/> 
    ```
    - If the mechanism has a closed kinematics (there is `<equality>` in the code), you need transform it to open kinematics.

    - Specify a `type` for each `<geom>`.

    - The mesh folder should contain only the files used in the model.

2) Make a simplified `.XML` hand model with capsules instead of meshes for contact links.

    Instead of this:
    ``` bash
    <geom quat="1 1 0 0" type="mesh" mesh="tongue"pos="0.05 -0.1 -0.009" />
    ```

    Do this:
    ``` bash
    <geom quat="1 1 0 0" type="capsule" size="0.012 0.008" pos="-0.005 0.025 -0.009"/>
    ```
    > The size parameter Sets the radius and half the length of the cylindrical part of the capsule (excluding the hemispheres at the ends).

3) Make a file `.JSON` with contact points on the links contact surfaces. Leave empty lists for links that are not involved in the contact. The list consists of `<body>` names.

    - File name `contact_points_{HAND_NAME}.json`

    - If the palm is large, contact points are also needed there.

    File Structure:

    ``` bash
    {
        "palm_link":[[x,y,z],[x,y,z]...,[x,y,z]],
        "finger_1_link_child":[[x,y,z],[x,y,z]...,[x,y,z]],
        "finger_2_link_child":[], # not involved in the contact
        ...
        "finger_N_link_child":[[x,y,z],[x,y,z]...,[x,y,z]]
    }
    ```

4) Make a file `.JSON` with penetration points on the links contact surfaces. The list consists of `<body>` names.

    - File name `penetration_points_{HAND_NAME}.json`

    File Structure:

    ``` bash 
    {
        "palm_link":[],
        "finger_1_link_child":[[x,y,z],[x,y,z]...,[x,y,z]],
        ...
        "finger_N_link_child":[[x,y,z],[x,y,z]...,[x,y,z]]
    }
    ```

5) Make a file `.JSON` with hand parameters:

     - File name `HAND_NAME.json`

     File Structure:

     ``` bash
    "joint_names": ["joint_1", ... "joint_N"],
    "init_pos": [0,0,0,0,0,0], # the initial positions of the joints, equal to the number of joint_names
    "face_verts_bodies": ["finger_1_link_child", ... "finger_N_link_child"], # links are in contact
    "ignore_bodies": ["palm_link", ...], # links are not in contact
    "radius": [0.018, 0.0145, 0.011, 0.018, 0.0145, 0.011, 0.018, 0.0145, 0.011] # ???
     ```