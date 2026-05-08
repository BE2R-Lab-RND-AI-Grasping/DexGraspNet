import argparse
from pathlib import Path
import sys

import numpy as np
import plotly.graph_objects as go
import torch
import transforms3d
import trimesh as tm

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.optimizer import TabletopOrientationBounds


TRANSLATION_KEYS = ("WRJTx", "WRJTy", "WRJTz")
ROTATION_KEYS = ("WRJRx", "WRJRy", "WRJRz")


def load_grasp_records(path):
    data = np.load(path, allow_pickle=True)
    return [item.item() if hasattr(item, "item") else item for item in data]


def qpos_to_translation_and_z_axis(qpos):
    translation = np.array([qpos[key] for key in TRANSLATION_KEYS], dtype=np.float64)
    euler = [qpos[key] for key in ROTATION_KEYS]
    rotation = transforms3d.euler.euler2mat(*euler, axes="sxyz")
    z_axis = rotation[:, 2]
    z_axis = z_axis / np.linalg.norm(z_axis)
    return translation, z_axis


def qpos_to_hand_pose_9d(qpos):
    translation = np.array([qpos[key] for key in TRANSLATION_KEYS], dtype=np.float64)
    euler = [qpos[key] for key in ROTATION_KEYS]
    rotation = transforms3d.euler.euler2mat(*euler, axes="sxyz")
    rot6d = rotation.T[:2].reshape(6)
    return np.concatenate([translation, rot6d])


def angle_to_world_xy_degrees(z_axis):
    return np.degrees(np.arcsin(np.clip(abs(z_axis[2]), 0.0, 1.0)))


def valid_mask_from_tabletop_bounds(records, pose_key, max_angle_degrees):
    hand_pose = np.stack([qpos_to_hand_pose_9d(record[pose_key]) for record in records], axis=0)
    hand_pose = torch.tensor(hand_pose, dtype=torch.float)
    bounds = TabletopOrientationBounds(max_angle_degrees=max_angle_degrees, device="cpu")
    return bounds.valid_mask(hand_pose).cpu().numpy().astype(bool)


def build_vector_traces(records, pose_key, max_count, stride, vector_length, max_angle_degrees, orientation_filter):
    candidate_records = records[::stride]
    if max_count is not None:
        candidate_records = candidate_records[:max_count]

    positions = []
    axes = []
    angles = []
    energies = []
    indices = []

    for pose_i, record in enumerate(candidate_records):
        translation, z_axis = qpos_to_translation_and_z_axis(record[pose_key])
        positions.append(translation)
        axes.append(z_axis)
        angles.append(angle_to_world_xy_degrees(z_axis))
        energies.append(record.get("energy", np.nan))
        indices.append(pose_i * stride)

    if not positions:
        raise ValueError("No poses selected. Try a larger --max-count or smaller --stride.")

    positions = np.stack(positions, axis=0)
    axes = np.stack(axes, axis=0)
    angles = np.array(angles)
    energies = np.array(energies)
    indices = np.array(indices)
    valid = valid_mask_from_tabletop_bounds(candidate_records, pose_key, max_angle_degrees)

    if orientation_filter == "valid":
        keep = valid
    elif orientation_filter == "invalid":
        keep = ~valid
    else:
        keep = np.ones_like(valid, dtype=bool)

    positions = positions[keep]
    axes = axes[keep]
    angles = angles[keep]
    energies = energies[keep]
    indices = indices[keep]
    valid = valid[keep]

    if len(positions) == 0:
        raise ValueError(f"No poses matched --orientation_filter {orientation_filter!r}.")

    hover_text = [
        f"pose={pose_i}<br>angle_to_xy={angle:.2f} deg<br>energy={energy:.4g}"
        for pose_i, angle, energy in zip(indices, angles, energies)
    ]

    marker_trace = go.Scatter3d(
        x=positions[:, 0],
        y=positions[:, 1],
        z=positions[:, 2],
        mode="markers",
        marker=dict(
            size=4,
            color=angles,
            colorscale="Viridis",
            colorbar=dict(title="angle to XY"),
            cmin=0,
            cmax=max(max_angle_degrees, float(np.max(angles))),
            symbol=np.where(valid, "circle", "x"),
        ),
        text=hover_text,
        hoverinfo="text",
        name="wrist translation",
    )

    line_x = []
    line_y = []
    line_z = []
    line_text = []
    for position, axis, text in zip(positions, axes, hover_text):
        end = position + vector_length * axis
        line_x.extend([position[0], end[0], None])
        line_y.extend([position[1], end[1], None])
        line_z.extend([position[2], end[2], None])
        line_text.extend([text, text, None])

    vector_trace = go.Scatter3d(
        x=line_x,
        y=line_y,
        z=line_z,
        mode="lines",
        line=dict(color="green", width=4),
        text=line_text,
        hoverinfo="text",
        name="wrist local Z",
    )

    return [marker_trace, vector_trace], {
        "candidates": len(candidate_records),
        "selected": len(positions),
        "valid": int(np.count_nonzero(valid)),
        "invalid": int(np.count_nonzero(~valid)),
        "min_angle": float(np.min(angles)),
        "max_angle": float(np.max(angles)),
    }


def object_mesh_trace(data_root_path, object_code, scale, opacity):
    mesh_path = Path(data_root_path) / object_code / "coacd" / "decomposed.obj"
    if not mesh_path.exists():
        raise FileNotFoundError(f"Object mesh not found: {mesh_path}")

    mesh = tm.load(mesh_path, force="mesh", process=False)
    vertices = np.asarray(mesh.vertices) * scale
    faces = np.asarray(mesh.faces)
    return go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        color="lightgreen",
        opacity=opacity,
        name=object_code,
    )


def collect_pose_files(result_path, hand_name, object_code):
    grasp_dir = Path(result_path) / hand_name
    if object_code:
        path = grasp_dir / f"{object_code}.npy"
        if not path.exists():
            raise FileNotFoundError(f"Pose file not found: {path}")
        return [path]

    paths = sorted(grasp_dir.glob("*.npy"))
    if not paths:
        raise FileNotFoundError(f"No .npy grasp files found in {grasp_dir}")
    return paths


def build_figure_for_file(path, args):
    object_code = path.stem
    records = load_grasp_records(path)
    traces = []

    if args.show_object:
        scale = records[0].get("scale", 1.0)
        traces.append(object_mesh_trace(args.data_root_path, object_code, scale, args.object_opacity))

    vector_traces, stats = build_vector_traces(
        records=records,
        pose_key=args.pose_key,
        max_count=args.max_count,
        stride=args.stride,
        vector_length=args.vector_length,
        max_angle_degrees=args.max_angle_degrees,
        orientation_filter=args.orientation_filter,
    )
    traces.extend(vector_traces)

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=(
            f"{object_code}: {args.pose_key} wrist Z vectors "
            f"({stats['valid']} valid / {stats['selected']} shown, "
            f"angle range {stats['min_angle']:.1f}-{stats['max_angle']:.1f} deg)"
        ),
        scene=dict(
            aspectmode="data",
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
        ),
        legend=dict(itemsizing="constant"),
    )
    return fig, stats


def output_path_for(input_path, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{input_path.stem}_z_vectors.html"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize generated wrist local-Z orientation vectors from saved grasp .npy files."
    )
    parser.add_argument("--result_path", default="../data/daily_props_decomposed_1000")
    parser.add_argument("--data_root_path", default="../data/daily_props_decomposed")
    parser.add_argument("--hand_name", default="hand_camera_efim")
    parser.add_argument("--object_code", default=None, help="Object code without .npy. If omitted, process all files.")
    parser.add_argument("--pose_key", choices=("qpos", "qpos_st"), default="qpos")
    parser.add_argument("--max_count", type=int, default=250, help="Maximum number of poses shown per object.")
    parser.add_argument("--stride", type=int, default=1, help="Use every Nth pose before max_count is applied.")
    parser.add_argument("--vector_length", type=float, default=0.1)
    parser.add_argument("--max_angle_degrees", type=float, default=37.0)
    parser.add_argument("--orientation_filter", choices=("all", "valid", "invalid"), default="all")
    parser.add_argument("--no_object", action="store_true", help="Do not draw the object mesh.")
    parser.add_argument("--object_opacity", type=float, default=0.45)
    parser.add_argument("--output_dir", default="visualizations/z_vectors")
    parser.add_argument("--show", action="store_true", help="Open the plot in a browser in addition to writing HTML.")
    return parser.parse_args()


def main():
    args = parse_args()
    args.show_object = not args.no_object
    if args.stride < 1:
        raise ValueError("--stride must be >= 1")
    if args.max_count is not None and args.max_count < 1:
        raise ValueError("--max_count must be >= 1")

    pose_files = collect_pose_files(args.result_path, args.hand_name, args.object_code)
    for pose_file in pose_files:
        fig, stats = build_figure_for_file(pose_file, args)
        html_path = output_path_for(pose_file, args.output_dir)
        fig.write_html(html_path)
        if args.show:
            fig.show()
        print(
            f"Wrote {html_path} | shown={stats['selected']} "
            f"valid={stats['valid']} invalid={stats['invalid']}"
        )


if __name__ == "__main__":
    main()
