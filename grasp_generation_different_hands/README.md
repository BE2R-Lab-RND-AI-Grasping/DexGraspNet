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
