from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import trimesh


UP_VECTOR = np.array([0.0, 0.0, 1.0])


@dataclass(frozen=True)
class RoofDetectionConfig:
    """Tuning values for hackathon-scale architectural GLB files."""

    min_normal_z: float = 0.85
    min_area_m2: float = 1.0
    ground_clearance_ratio: float = 0.12
    coplanar_dot_threshold: float = 0.985
    plane_distance_m: float = 0.12
    max_roofs: int = 80


def process_glb_model(file_path: str | Path) -> dict[str, Any]:
    """
    Load a GLB model, segment planar facets, and return upward roof planes.

    The returned geometry is already remapped to per-roof local vertex indices
    so the frontend can build BufferGeometry directly.
    """

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file does not exist: {path}")

    mesh = _load_as_single_mesh(path)
    if mesh.faces is None or len(mesh.faces) == 0:
        raise ValueError("The uploaded GLB does not contain any triangle mesh faces.")

    if mesh.vertices is None or len(mesh.vertices) == 0:
        raise ValueError("The uploaded GLB does not contain mesh vertices.")

    mesh.remove_unreferenced_vertices()

    config = RoofDetectionConfig()
    roof_faces = _detect_roof_face_groups(mesh, config)
    detected_roof_count = len(roof_faces)
    roof_faces = sorted(roof_faces, key=lambda faces: float(mesh.area_faces[faces].sum()), reverse=True)
    roof_faces = roof_faces[: config.max_roofs]

    roofs = [_build_roof_payload(mesh, faces, index) for index, faces in enumerate(roof_faces, start=1)]
    roofs.sort(key=lambda roof: roof["area_m2"], reverse=True)

    return {
        "roof_count": len(roofs),
        "detected_roof_count": detected_roof_count,
        "total_roof_area_m2": round(float(sum(roof["area_m2"] for roof in roofs)), 3),
        "returned_roof_limit": config.max_roofs,
        "roofs": roofs,
        "model_bounds": {
            "min": _round_vector(mesh.bounds[0]),
            "max": _round_vector(mesh.bounds[1]),
        },
    }


def _load_as_single_mesh(path: Path) -> trimesh.Trimesh:
    if _glb_uses_draco(path):
        return _load_draco_glb_as_mesh(path)

    loaded = trimesh.load(path, force="scene", process=False)

    if isinstance(loaded, trimesh.Trimesh):
        return loaded

    if isinstance(loaded, trimesh.Scene):
        try:
            dumped = loaded.dump(concatenate=True)
        except BaseException as exc:
            raise ValueError(f"Could not flatten GLB scene into one mesh: {exc}") from exc

        if isinstance(dumped, trimesh.Trimesh):
            return dumped

        if isinstance(dumped, (list, tuple)) and dumped:
            meshes = [item for item in dumped if isinstance(item, trimesh.Trimesh) and len(item.faces) > 0]
            if meshes:
                return trimesh.util.concatenate(meshes)

    raise ValueError("Could not find usable mesh geometry in the uploaded GLB.")


def _glb_uses_draco(path: Path) -> bool:
    try:
        from pygltflib import GLTF2

        gltf = GLTF2().load(str(path))
    except BaseException:
        return False

    extensions_required = set(gltf.extensionsRequired or [])
    extensions_used = set(gltf.extensionsUsed or [])
    return "KHR_draco_mesh_compression" in extensions_required | extensions_used


def _load_draco_glb_as_mesh(path: Path) -> trimesh.Trimesh:
    try:
        import DracoPy
        from pygltflib import GLTF2
    except ImportError as exc:
        raise ValueError(
            "This GLB uses KHR_draco_mesh_compression. Install DracoPy and pygltflib to decode it."
        ) from exc

    gltf = GLTF2().load(str(path))
    blob = gltf.binary_blob()
    if not blob:
        raise ValueError("The Draco-compressed GLB does not contain a binary buffer.")

    decoded_meshes: list[trimesh.Trimesh] = []

    def decode_mesh(mesh_index: int, transform: np.ndarray) -> None:
        gltf_mesh = gltf.meshes[mesh_index]
        for primitive in gltf_mesh.primitives:
            extension = (primitive.extensions or {}).get("KHR_draco_mesh_compression")
            if not extension:
                continue

            buffer_view_index = extension.get("bufferView")
            if buffer_view_index is None:
                continue

            buffer_view = gltf.bufferViews[buffer_view_index]
            start = buffer_view.byteOffset or 0
            end = start + buffer_view.byteLength
            draco_buffer = blob[start:end]
            draco_mesh = DracoPy.decode(draco_buffer)

            points = getattr(draco_mesh, "points", None)
            faces = getattr(draco_mesh, "faces", None)
            if points is None or faces is None or len(points) == 0 or len(faces) == 0:
                continue

            mesh = trimesh.Trimesh(
                vertices=np.asarray(points, dtype=np.float64),
                faces=np.asarray(faces, dtype=np.int64),
                process=False,
            )
            mesh.apply_transform(transform)
            decoded_meshes.append(mesh)

    def visit_node(node_index: int, parent_transform: np.ndarray) -> None:
        node = gltf.nodes[node_index]
        transform = parent_transform @ _node_transform(node)

        if node.mesh is not None:
            decode_mesh(node.mesh, transform)

        for child_index in node.children or []:
            visit_node(child_index, transform)

    scene_index = gltf.scene if gltf.scene is not None else 0
    if gltf.scenes and gltf.scenes[scene_index].nodes:
        for node_index in gltf.scenes[scene_index].nodes:
            visit_node(node_index, np.eye(4))
    else:
        for mesh_index in range(len(gltf.meshes or [])):
            decode_mesh(mesh_index, np.eye(4))

    if not decoded_meshes:
        raise ValueError("Could not decode any Draco mesh primitives from the uploaded GLB.")

    return trimesh.util.concatenate(decoded_meshes)


def _node_transform(node: Any) -> np.ndarray:
    if node.matrix:
        return np.asarray(node.matrix, dtype=np.float64).reshape((4, 4)).T

    transform = np.eye(4)

    if node.translation:
        transform[:3, 3] = np.asarray(node.translation, dtype=np.float64)

    if node.rotation:
        quaternion = np.asarray(node.rotation, dtype=np.float64)
        transform = transform @ trimesh.transformations.quaternion_matrix(
            [quaternion[3], quaternion[0], quaternion[1], quaternion[2]]
        )

    if node.scale:
        scale = np.asarray(node.scale, dtype=np.float64)
        transform = transform @ np.diag([scale[0], scale[1], scale[2], 1.0])

    return transform


def _detect_roof_face_groups(mesh: trimesh.Trimesh, config: RoofDetectionConfig) -> list[np.ndarray]:
    z_min = float(mesh.bounds[0][2])
    z_max = float(mesh.bounds[1][2])
    height = max(z_max - z_min, 1e-6)
    min_center_z = z_min + (height * config.ground_clearance_ratio)

    # Avoid trimesh.facets here: on large Draco/Cesium building exports it can
    # spend a long time building global coplanar groups. Local adjacency gives
    # the roof patches we need and keeps uploads responsive.
    upward_faces = []
    for face_index, normal in enumerate(mesh.face_normals):
        adjusted_normal = _average_normal(np.asarray([normal]))
        if float(np.dot(adjusted_normal, UP_VECTOR)) > config.min_normal_z:
            face_center_z = float(mesh.triangles_center[face_index][2])
            if face_center_z > min_center_z and float(mesh.area_faces[face_index]) >= 1e-6:
                upward_faces.append(face_index)

    groups = _merge_adjacent_coplanar_faces(mesh, upward_faces, config)
    return [group for group in groups if _is_roof_candidate(mesh, group, min_center_z, config)]


def _is_roof_candidate(
    mesh: trimesh.Trimesh,
    face_indices: np.ndarray,
    min_center_z: float,
    config: RoofDetectionConfig,
) -> bool:
    if len(face_indices) == 0:
        return False

    normal = _average_normal(mesh.face_normals[face_indices])
    area = float(mesh.area_faces[face_indices].sum())
    center_z = float(_weighted_face_center(mesh, face_indices)[2])

    return (
        float(np.dot(normal, UP_VECTOR)) > config.min_normal_z
        and area >= config.min_area_m2
        and center_z > min_center_z
    )


def _merge_adjacent_coplanar_faces(
    mesh: trimesh.Trimesh,
    face_indices: list[int],
    config: RoofDetectionConfig,
) -> list[np.ndarray]:
    if not face_indices:
        return []

    allowed = set(face_indices)
    parent = {face_index: face_index for face_index in face_indices}

    def find(face_index: int) -> int:
        while parent[face_index] != face_index:
            parent[face_index] = parent[parent[face_index]]
            face_index = parent[face_index]
        return face_index

    def union(a: int, b: int) -> None:
        root_a = find(a)
        root_b = find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    for face_a, face_b in mesh.face_adjacency:
        face_a = int(face_a)
        face_b = int(face_b)
        if face_a not in allowed or face_b not in allowed:
            continue

        normal_a = _average_normal(np.asarray([mesh.face_normals[face_a]]))
        normal_b = _average_normal(np.asarray([mesh.face_normals[face_b]]))
        same_direction = float(np.dot(normal_a, normal_b)) > config.coplanar_dot_threshold
        center_delta = mesh.triangles_center[face_a] - mesh.triangles_center[face_b]
        same_plane = abs(float(np.dot(center_delta, normal_a))) < config.plane_distance_m

        if same_direction and same_plane:
            union(face_a, face_b)

    grouped: dict[int, list[int]] = {}
    for face_index in face_indices:
        grouped.setdefault(find(face_index), []).append(face_index)

    groups = [np.asarray(group, dtype=np.int64) for group in grouped.values()]
    return [group for group in groups if float(mesh.area_faces[group].sum()) >= config.min_area_m2]


def _build_roof_payload(mesh: trimesh.Trimesh, face_indices: np.ndarray, roof_id: int) -> dict[str, Any]:
    selected_faces = mesh.faces[face_indices]
    unique_vertex_ids, inverse = np.unique(selected_faces.reshape(-1), return_inverse=True)
    remapped_indices = inverse.reshape((-1, 3))
    vertices = mesh.vertices[unique_vertex_ids]

    normal = _average_normal(mesh.face_normals[face_indices])
    center = _weighted_face_center(mesh, face_indices)
    dimensions = _roof_dimensions(vertices, normal)
    area = float(mesh.area_faces[face_indices].sum())

    # Raise overlay and labels slightly along the roof normal to avoid z-fighting.
    overlay_offset = 0.03
    label_offset = 0.35

    return {
        "id": roof_id,
        "vertices": [_round_vector(vertex + normal * overlay_offset) for vertex in vertices],
        "indices": remapped_indices.astype(int).reshape(-1).tolist(),
        "normal": _round_vector(normal),
        "center": _round_vector(center),
        "label_position": _round_vector(center + normal * label_offset),
        "length_m": round(dimensions["length"], 3),
        "width_m": round(dimensions["width"], 3),
        "area_m2": round(area, 3),
    }


def _roof_dimensions(vertices: np.ndarray, normal: np.ndarray) -> dict[str, float]:
    if len(vertices) < 3:
        return {"length": 0.0, "width": 0.0}

    centered = vertices - vertices.mean(axis=0)

    try:
        _, extents = trimesh.bounds.oriented_bounds(vertices)
        positive_extents = sorted((float(value) for value in extents if value > 1e-6), reverse=True)
        if len(positive_extents) >= 2:
            return {"length": positive_extents[0], "width": positive_extents[1]}
    except BaseException:
        pass

    # PCA fallback projected into the detected roof plane.
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis_a = _project_to_plane(vh[0], normal)
    if np.linalg.norm(axis_a) < 1e-8:
        axis_a = _orthogonal_axis(normal)
    axis_a = axis_a / np.linalg.norm(axis_a)
    axis_b = np.cross(normal, axis_a)
    axis_b = axis_b / np.linalg.norm(axis_b)

    projected_a = centered @ axis_a
    projected_b = centered @ axis_b

    extents = sorted(
        [
            float(projected_a.max() - projected_a.min()),
            float(projected_b.max() - projected_b.min()),
        ],
        reverse=True,
    )
    return {"length": extents[0], "width": extents[1]}


def _average_normal(normals: np.ndarray) -> np.ndarray:
    normal = np.mean(normals, axis=0)
    norm = float(np.linalg.norm(normal))
    if norm < 1e-8:
        return UP_VECTOR.copy()

    normal = normal / norm
    if normal[2] < 0:
        normal = -normal
    return normal


def _weighted_face_center(mesh: trimesh.Trimesh, face_indices: np.ndarray) -> np.ndarray:
    centers = mesh.triangles_center[face_indices]
    weights = mesh.area_faces[face_indices]
    total_weight = float(weights.sum())
    if total_weight <= 1e-12:
        return centers.mean(axis=0)
    return np.average(centers, axis=0, weights=weights)


def _project_to_plane(vector: np.ndarray, normal: np.ndarray) -> np.ndarray:
    return vector - normal * float(np.dot(vector, normal))


def _orthogonal_axis(normal: np.ndarray) -> np.ndarray:
    reference = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(reference, normal))) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    return _project_to_plane(reference, normal)


def _round_vector(vector: np.ndarray) -> list[float]:
    return [round(float(value), 6) for value in vector]
