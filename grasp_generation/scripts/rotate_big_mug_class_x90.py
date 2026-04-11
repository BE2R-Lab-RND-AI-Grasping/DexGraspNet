"""Rotate all meshes under BIG_MUG_CLASS by +90 degrees around the X axis."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import trimesh


SUPPORTED_EXTS = {".obj", ".stl"}


def rotation_matrix_x(degrees: float) -> np.ndarray:
    radians = math.radians(degrees)
    cos_t = math.cos(radians)
    sin_t = math.sin(radians)
    return np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, cos_t, -sin_t, 0.0],
            [0.0, sin_t, cos_t, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def rotate_mesh_file(
    path: Path,
    transform: np.ndarray,
    output_root: Path | None,
    root: Path,
) -> Path:
    mesh = trimesh.load_mesh(path, force="mesh")
    mesh.apply_transform(transform)

    if output_root is None:
        output_path = path
    else:
        relative = path.relative_to(root)
        output_path = output_root / relative
        output_path.parent.mkdir(parents=True, exist_ok=True)

    mesh.export(output_path)
    return output_path


def iter_mesh_files(root: Path) -> list[Path]:
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rotate all mesh files under   by +90 degrees around X axis.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("../data/BIG_BOWLS"),
        help="Path to   folder.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Optional output directory. If omitted, files are overwritten in place.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print files that would be rotated without writing changes.",
    )
    args = parser.parse_args()

    root = args.root
    if not root.exists():
        raise FileNotFoundError(f"Root path not found: {root}")

    transform = rotation_matrix_x(90.0)
    mesh_files = iter_mesh_files(root)
    if not mesh_files:
        print(f"No mesh files found under {root}")
        return

    for mesh_path in mesh_files:
        if args.dry_run:
            print(f"[dry-run] would rotate {mesh_path}")
            continue
        output_path = rotate_mesh_file(mesh_path, transform, args.output_root, root)
        print(f"rotated {mesh_path} -> {output_path}")


if __name__ == "__main__":
    main()