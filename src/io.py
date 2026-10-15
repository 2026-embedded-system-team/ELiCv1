"""Point-cloud I/O without optional visualization dependencies."""

import os
from concurrent.futures import ThreadPoolExecutor
from itertools import islice

import numpy as np


def _ply_vertex_layout(stream):
    """Read an ASCII PLY header and locate its scalar XYZ vertex columns."""
    if stream.readline().strip() != "ply":
        raise ValueError("Expected a PLY file beginning with 'ply'")

    ascii_format = False
    elements = []
    current = None
    while True:
        line = stream.readline()
        if not line:
            raise ValueError("PLY header is missing end_header")
        fields = line.split()
        if not fields or fields[0] in ("comment", "obj_info"):
            continue
        tag = fields[0]
        if tag == "end_header":
            if len(fields) != 1:
                raise ValueError("Malformed PLY end_header")
            break
        if tag == "format":
            if fields != ["format", "ascii", "1.0"] or ascii_format:
                raise ValueError("Only PLY format ascii 1.0 is supported")
            ascii_format = True
        elif tag == "element":
            if len(fields) != 3:
                raise ValueError("Malformed PLY element declaration")
            try:
                count = int(fields[2])
            except ValueError as exc:
                raise ValueError("PLY element count must be an integer") from exc
            if count < 0:
                raise ValueError("PLY element count must be nonnegative")
            current = (fields[1], count, [])
            elements.append(current)
        elif tag == "property":
            if current is None:
                raise ValueError("PLY property has no element")
            if current[0] == "vertex":
                if len(fields) != 3 or fields[1] == "list":
                    raise ValueError("PLY vertex properties must be scalar")
                current[2].append(fields[2])
        else:
            raise ValueError(f"Unsupported PLY header declaration: {tag}")

    if not ascii_format:
        raise ValueError("PLY header is missing format ascii 1.0")
    vertices = [element for element in elements if element[0] == "vertex"]
    if len(vertices) != 1:
        raise ValueError("PLY header must declare exactly one vertex element")
    _, count, properties = vertices[0]
    if any(properties.count(axis) != 1 for axis in ("x", "y", "z")):
        raise ValueError("PLY vertex properties must include x, y, and z exactly once")
    preceding_rows = 0
    for name, size, _ in elements:
        if name == "vertex":
            break
        preceding_rows += size
    return count, tuple(properties.index(axis) for axis in ("x", "y", "z")), preceding_rows


def read_points(filedir):
    """Read XYZ from an ASCII PLY or float32 XYZI binary file.

    PLY results remain float64, as in the original implementation. Binary
    results remain float32. Other PLY elements and extra scalar properties
    are excluded using the header, rather than interpreted as extra points.
    """
    if os.path.splitext(os.fspath(filedir))[1].lower() == ".bin":
        if os.path.getsize(filedir) % (4 * np.dtype(np.float32).itemsize):
            raise ValueError("Binary point cloud must contain float32 XYZI records")
        values = np.fromfile(filedir, dtype=np.float32)
        if values.size % 4:
            raise ValueError("Binary point cloud must contain float32 XYZI records")
        return values.reshape(-1, 4)[:, :3]

    with open(filedir, "r", encoding="utf-8") as stream:
        count, xyz_columns, preceding_rows = _ply_vertex_layout(stream)
        for _ in range(preceding_rows):
            if not stream.readline():
                raise ValueError("PLY data is truncated before the vertex element")
        if count == 0:
            return np.empty((0, 3), dtype=np.float64)
        # Limit physical rows so face/edge records cannot become extra points.
        points = np.loadtxt(
            islice(stream, count), dtype=np.float64,
            usecols=xyz_columns, comments=None, ndmin=2,
        )
        if points.shape != (count, 3):
            raise ValueError(f"PLY vertex count mismatch: expected {count}, got {len(points)}")
        return points


def read_point_clouds(file_path_list, workers=None):
    """Read files in input order with a bounded, Windows-safe thread pool.

    ``workers=1`` selects sequential reading. The default uses at most eight
    workers and never starts more workers than files or available CPUs.
    Threads avoid process-spawn recursion when callers run as CLI scripts.
    """
    paths = list(file_path_list)
    if workers is not None and (not isinstance(workers, int) or workers < 1):
        raise ValueError("workers must be a positive integer or None")
    if not paths:
        return []
    limit = min(len(paths), os.cpu_count() or 1, 8) if workers is None else min(workers, len(paths))
    if limit == 1:
        return [read_points(path) for path in paths]
    with ThreadPoolExecutor(max_workers=limit) as pool:
        return list(pool.map(read_points, paths))


def save_ply_ascii_geo(coords, filedir):
    """Overwrite an ASCII PLY, preserving every input value after float32 cast."""
    points = np.asarray(coords, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("coords must have shape (N, 3)")
    with open(filedir, "w", encoding="ascii", newline="\n") as stream:
        stream.write(
            "ply\nformat ascii 1.0\n"
            f"element vertex {len(points)}\n"
            "property float x\nproperty float y\nproperty float z\nend_header\n"
        )
        # Nine significant digits round-trip all finite float32 values.
        np.savetxt(stream, points, fmt="%.9g")
