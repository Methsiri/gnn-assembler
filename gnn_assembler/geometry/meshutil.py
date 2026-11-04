"""Triangle meshes for box parts, distances, and file IO."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh


def box_mesh(low: np.ndarray, high: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x0, y0, z0 = np.asarray(low, dtype=float)
    x1, y1, z1 = np.asarray(high, dtype=float)
    vertices = np.array(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ],
        dtype=float,
    )
    quads = (
        (0, 2, 1),
        (0, 3, 2),
        (4, 5, 6),
        (4, 6, 7),
        (0, 1, 5),
        (0, 5, 4),
        (1, 2, 6),
        (1, 6, 5),
        (2, 3, 7),
        (2, 7, 6),
        (3, 0, 4),
        (3, 4, 7),
    )
    return vertices, np.array(quads, dtype=np.int64)


def mesh_from_boxes(bounds: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    offset = 0
    for low, high in bounds:
        verts, tris = box_mesh(low, high)
        vertices.append(verts)
        faces.append(tris + offset)
        offset += len(verts)
    return np.vstack(vertices), np.vstack(faces)


def load_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    loaded = trimesh.load(path, force="mesh", process=False)
    if not isinstance(loaded, trimesh.Trimesh):
        raise ValueError(f"Expected a triangle mesh at {path}")
    if loaded.vertices.size == 0 or loaded.faces.size == 0:
        raise ValueError(f"Mesh {path} has no triangles")
    return np.asarray(loaded.vertices, dtype=float), np.asarray(loaded.faces, dtype=np.int64)


def save_mesh(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.export(path)


def point_triangle_distance(point: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Closest distance from a point to a triangle (Ericson)."""

    ab = b - a
    ac = c - a
    ap = point - a
    d1 = float(np.dot(ab, ap))
    d2 = float(np.dot(ac, ap))
    if d1 <= 0.0 and d2 <= 0.0:
        return float(np.linalg.norm(ap))
    bp = point - b
    d3 = float(np.dot(ab, bp))
    d4 = float(np.dot(ac, bp))
    if d3 >= 0.0 and d4 <= d3:
        return float(np.linalg.norm(bp))
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        denom = d1 - d3
        v = d1 / denom if denom != 0.0 else 0.0
        return float(np.linalg.norm(point - (a + v * ab)))
    cp = point - c
    d5 = float(np.dot(ab, cp))
    d6 = float(np.dot(ac, cp))
    if d6 >= 0.0 and d5 <= d6:
        return float(np.linalg.norm(cp))
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        denom = d2 - d6
        w = d2 / denom if denom != 0.0 else 0.0
        return float(np.linalg.norm(point - (a + w * ac)))
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        denom = (d4 - d3) + (d5 - d6)
        w = (d4 - d3) / denom if denom != 0.0 else 0.0
        return float(np.linalg.norm(point - (b + w * (c - b))))
    denom = va + vb + vc
    if denom == 0.0:
        return float(np.linalg.norm(ap))
    v = vb / denom
    w = vc / denom
    return float(np.linalg.norm(point - (a + ab * v + ac * w)))


def mesh_distance(
    vertices_a: np.ndarray,
    faces_a: np.ndarray,
    vertices_b: np.ndarray,
    faces_b: np.ndarray,
) -> float:
    best = float("inf")
    for point in vertices_a:
        for tri in faces_b:
            best = min(
                best,
                point_triangle_distance(point, vertices_b[tri[0]], vertices_b[tri[1]], vertices_b[tri[2]]),
            )
            if best <= 1e-8:
                return 0.0
    for point in vertices_b:
        for tri in faces_a:
            best = min(
                best,
                point_triangle_distance(point, vertices_a[tri[0]], vertices_a[tri[1]], vertices_a[tri[2]]),
            )
            if best <= 1e-8:
                return 0.0
    return best
