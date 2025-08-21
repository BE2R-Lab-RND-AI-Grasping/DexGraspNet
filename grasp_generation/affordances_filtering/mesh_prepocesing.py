import numpy as np
import trimesh

def find_contact_points(
    hand_mesh: trimesh.Trimesh,
    obj_mesh: trimesh.Trimesh,
    contact_thr=0.0025,
    num_obj_point=5000,
):
    """
    This function samples points from the object mesh and finds the closest points on the hand mesh.
    Args:
        hand_mesh (trimesh.Trimesh): The mesh representing the hand
        obj_mesh (trimesh.Trimesh): The mesh representing the object
        contact_thr (float, optional): Distance threshold to consider a point as a contact point. Defaults to 0.0025.
        num_obj_point (int, optional): Number of points to sample from the object surface. Defaults to 5000.
    Returns:
        numpy.ndarray: Array of contact points on the hand mesh that are within the threshold distance from the object
    """
    points = trimesh.sample.sample_surface(obj_mesh, count=num_obj_point)
    closest_points, distances, triangle_ids = trimesh.proximity.closest_point(hand_mesh, points[0])
 
    filtered_points = closest_points[distances < contact_thr]
    return filtered_points


def create_affordance_mesh_from_color(object_scene,
    affordance_color = [255, 0, 0, 255]
):
    affordance_color_numpy = np.array(affordance_color, dtype=np.uint8)
    if isinstance(object_scene, trimesh.Trimesh):
        mesh_vis_color = object_scene
    elif isinstance(object_scene, trimesh.Scene):
        mesh_vis_color = colored_scene2mesh(object_scene)   
    else:
        raise ValueError("object_scene must be a trimesh.Trimesh or trimesh.Scene instance")
    vertex_colors = mesh_vis_color.visual.vertex_colors
    faces_colors = mesh_vis_color.visual.face_colors

    # Find rows where any value in the row is in `values`
    matches_vertex = np.all(np.equal(vertex_colors, affordance_color_numpy), axis=1)
    matches_face = np.all(np.equal(faces_colors, affordance_color_numpy), axis=1)

    indices_vertices = np.where(matches_vertex)[0]
    indices_faces = np.where(matches_face)[0]
    
    affordances_vertices = mesh_vis_color.vertices[indices_vertices]
    affordances_faces = mesh_vis_color.faces[indices_faces]
    affordances_meshes = trimesh.Trimesh(affordances_vercticals, affordances_faces)
    return affordances_meshes

def colored_scene2mesh(object_scene):
    mesh_vis_color = object_scene.to_mesh()
    
    # Check if visual needs conversion or is already a ColorVisuals object
    if hasattr(mesh_vis_color.visual, 'to_color'):
        visual = mesh_vis_color.visual.to_color()
        mesh_vis_color.visual = visual
    else:
        raise FileExistsError(f"Obviosly, you need to add .mtl file to the scene: "
                              f"{object_scene.metadata.get('file_name', 'unknown')}")
    return mesh_vis_color

def filter_contact_points(affordance_mesh: trimesh.Trimesh, contact_points: np.ndarray, incontact_thr=0.0025):
    closest_points, distances, triangle_ids = trimesh.proximity.closest_point(affordance_mesh, contact_points)
    filtred_points = closest_points[distances < incontact_thr]
    return filtred_points

def get_affordance_contact_points(obj_mesh, 
                                  hand_mesh,
                                  num_obj_point=1000, 
                                  incontact_distance_thr=0.0025,
                                  min_contact_point = 10,
                                  affordance_color=[255, 0, 0, 255]):
    """
    Extracts affordance contact points between an object and a hand mesh.
    This function identifies contact points between a hand mesh and an object mesh, 
    then filters these points to find those located on the affordance regions 
    of the object (defined by a specific color).
    Parameters
    ----------
    obj_mesh : trimesh.Trimesh
        The object mesh to analyze for contact points.
    hand_mesh : trimesh.Trimesh
        The hand mesh to find contact points with the object.
    num_obj_point : int, default=1000
        Number of sample points to use on the object mesh.
    incontact_distance_thr : float, default=0.0025
        Distance threshold (in meters) to determine if points are in contact.
    min_contact_point : int, default=10
        The threshold at which it is considered that contact with an object has occurred.
    affordance_color : list, default=[255, 0, 0, 255]
        RGBA color values defining the affordance regions on the object mesh.
    Returns
    -------
    tuple
        A tuple containing two numpy arrays:
        - affordance_contact_points: Contact points located on affordance regions
        - contact_points: All contact points between the hand and object meshes
        hand_mesh: trimesh.Trimesh, 
    """
    affordance_contact_points = np.array([])
    contact_points = np.array([])

    contact_points = find_contact_points(hand_mesh, obj_mesh, num_obj_point=num_obj_point, contact_thr=incontact_distance_thr)
    affordance_mesh = create_affordance_mesh_from_color(obj_mesh, affordance_color)
    if len(contact_points) > min_contact_point:
        affordance_contact_points = filter_contact_points(affordance_mesh, contact_points)

    return affordance_contact_points, contact_points

def create_scene_contact_points(contact_points=None, affordance_contact_points=None, 
                            point_size=0.0025, contact_color=[255, 0, 255, 255], affordance_color=[0, 255, 0, 255]):

    scene = trimesh.Scene()
 
    # Add regular contact points if provided
    if contact_points is not None and len(contact_points) > 0:
        for point in contact_points:
            sphere = trimesh.creation.icosphere(radius=point_size)
            sphere.visual.face_colors = contact_color
            sphere.apply_translation(point)
            scene.add_geometry(sphere)
    
    # Add affordance contact points if provided
    if affordance_contact_points is not None and len(affordance_contact_points) > 0:
        for point in affordance_contact_points:
            sphere = trimesh.creation.icosphere(radius=point_size*1.2)
            sphere.visual.face_colors = affordance_color
            sphere.apply_translation(point)
            scene.add_geometry(sphere)
    
    return scene


def main():
    hand_mesh = trimesh.load("grasp_generation/test_meshes/hand_mesh_test.obj")
    # Required .mtl for adding colors to the mesh
    obj_scene = trimesh.load("grasp_generation/test_meshes/affordance_banana.obj")
    obj_mesh = colored_scene2mesh(obj_scene)
    obj_mesh.apply_scale(0.1)

    
    
    affordance_contact_points, contact_points = get_affordance_contact_points(obj_mesh, hand_mesh, 
                                                                            num_obj_point=1000, 
                                                                            incontact_distance_thr=0.0025, 
                                                                            min_contact_point=10)
    scene = create_scene_contact_points(contact_points=contact_points, 
                                    affordance_contact_points=affordance_contact_points, point_size=0.001)
    scene.add_geometry(obj_mesh)
    scene.add_geometry(hand_mesh)
    scene.show()

if __name__ == "__main__":
    main()