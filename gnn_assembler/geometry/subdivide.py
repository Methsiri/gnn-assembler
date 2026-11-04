"""Midpoint triangle subdivision for denser contact sampling."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import shutil

import numpy as np

from gnn_assembler.geometry.meshutil import load_mesh, save_mesh

MESH_SUFFIXES = {".obj", ".stl"}


def subdivide_once(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    verts = [np.asarray(row, dtype=float) for row in vertices]
    midpoint: dict[tuple[int, int], int] = {}

    def mid(i: int, j: int) -> int:
        key = (i, j) if i < j else (j, i)
        cached = midpoint.get(key)
        if cached is not None:
            return cached
        index = len(verts)
        verts.append((vertices[i] + vertices[j]) * 0.5)
        midpoint[key] = index
        return index

    new_faces: list[tuple[int, int, int]] = []
    for i, j, k in faces:
        a = mid(int(i), int(j))
        b = mid(int(j), int(k))
        c = mid(int(k), int(i))
        new_faces.extend(
            (
                (int(i), a, c),
                (int(j), b, a),
                (int(k), c, b),
                (a, b, c),
            )
        )
    return np.asarray(verts, dtype=float), np.asarray(new_faces, dtype=np.int64)


def subdivide_mesh(vertices: np.ndarray, faces: np.ndarray, levels: int) -> tuple[np.ndarray, np.ndarray]:
    if levels < 0:
        raise ValueError("levels must be >= 0")
    current_v, current_f = vertices, faces
    for _ in range(levels):
        current_v, current_f = subdivide_once(current_v, current_f)
    return current_v, current_f


def mesh_files(directory: Path) -> list[Path]:
    return sorted(
        path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in MESH_SUFFIXES
    )


def subdivide_directory(source: str | Path, target: str | Path, levels: int = 1) -> None:
    source_path = Path(source)
    target_path = Path(target)
    target_path.mkdir(parents=True, exist_ok=True)
    meta = source_path / "assembly.json"
    if meta.exists():
        shutil.copy(meta, target_path / "assembly.json")
    files = mesh_files(source_path)
    if not files and not meta.exists():
        raise FileNotFoundError(f"No meshes in {source_path}")
    for mesh_path in files:
        vertices, faces = load_mesh(mesh_path)
        dense_v, dense_f = subdivide_mesh(vertices, faces, levels)
        save_mesh(target_path / mesh_path.name, dense_v, dense_f)


def subdivide_batch(
    source: str | Path,
    target: str | Path,
    levels: int = 1,
    num_proc: int = 1,
) -> None:
    """Subdivide one assembly directory or each child assembly directory."""

    source_path = Path(source)
    target_path = Path(target)
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    if mesh_files(source_path) or (source_path / "assembly.json").exists():
        jobs = [(source_path, target_path)]
    else:
        children = sorted(path for path in source_path.iterdir() if path.is_dir())
        jobs = [(child, target_path / child.name) for child in children]
        if not jobs:
            raise FileNotFoundError(f"No assemblies under {source_path}")
    if num_proc <= 1 or len(jobs) == 1:
        for src, dst in jobs:
            subdivide_directory(src, dst, levels)
        return
    with ProcessPoolExecutor(max_workers=num_proc) as pool:
        futures = [pool.submit(subdivide_directory, src, dst, levels) for src, dst in jobs]
        for future in futures:
            future.result()
