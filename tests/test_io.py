import ast
from pathlib import Path

import numpy as np
import pytest

from src import io


ROOT = Path(__file__).resolve().parents[1]
SAMPLES = sorted((ROOT / "data").glob("*Example/*.ply"))


def original_read_points(path):
    """Execute the unchanged baseline reader without importing open3d."""
    source = ROOT / "benchmarks" / "reference" / "src" / "io.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "read_points")
    namespace = {"os": __import__("os"), "np": np}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["read_points"](path)


def write_ply(path, header, rows=""):
    path.write_text("ply\nformat ascii 1.0\n" + header + "end_header\n" + rows, encoding="utf-8")
    return path


@pytest.mark.parametrize("path", SAMPLES, ids=lambda path: str(path.relative_to(ROOT)))
def test_real_samples_match_original(path):
    result = io.read_points(path)
    np.testing.assert_array_equal(result, original_read_points(path))
    assert result.dtype == np.float64


def test_sample_inventory():
    assert len(SAMPLES) == 4


def test_named_xyz_properties_and_face_exclusion(tmp_path):
    path = write_ply(
        tmp_path / "extra properties.ply",
        "comment test\nelement vertex 2\nproperty uchar red\nproperty float z\n"
        "property double x\nproperty float y\nelement face 1\nproperty list uchar int vertex_indices\n",
        "255\t3 1 2\n10 6 4 5\n3 0 1 0\n",
    )
    np.testing.assert_array_equal(io.read_points(path), [[1, 2, 3], [4, 5, 6]])


def test_elements_before_vertex(tmp_path):
    path = write_ply(
        tmp_path / "before.ply",
        "element edge 1\nproperty int source\nproperty int target\n"
        "element vertex 1\nproperty float x\nproperty float y\nproperty float z\n",
        "0 1\n2 3 4\n",
    )
    np.testing.assert_array_equal(io.read_points(path), [[2, 3, 4]])


@pytest.mark.parametrize("count, rows, expected", [(0, "", np.empty((0, 3))), (1, "1 2 3\n", [[1, 2, 3]])])
def test_empty_and_single_vertex(tmp_path, count, rows, expected):
    path = write_ply(tmp_path / "points.ply", f"element vertex {count}\nproperty float x\nproperty float y\nproperty float z\n", rows)
    np.testing.assert_array_equal(io.read_points(path), expected)


@pytest.mark.parametrize("text", [
    "not ply\n", "ply\nformat ascii 1.0\n", "ply\nformat binary_little_endian 1.0\nend_header\n",
    "ply\nelement vertex 0\nproperty float x\nproperty float y\nproperty float z\nend_header\n",
    "ply\nformat ascii 1.0\nelement vertex -1\nend_header\n",
    "ply\nformat ascii 1.0\nelement vertex invalid\nend_header\n",
    "ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\nend_header\n1 2\n",
    "ply\nformat ascii 1.0\nelement vertex 1\nproperty list uchar float x\nend_header\n",
    "ply\nformat ascii 1.0\nelement vertex 2\nproperty float x\nproperty float y\nproperty float z\nend_header\n1 2 3\n",
])
def test_malformed_ply_rejected(tmp_path, text):
    path = tmp_path / "bad.ply"
    path.write_text(text)
    with pytest.raises(ValueError):
        io.read_points(path)


def test_binary_float32_xyzi(tmp_path):
    records = np.array([[1.25, -2.5, 3.75, 0.5], [9, 8, 7, 6]], dtype=np.float32)
    path = tmp_path / "raw points.BIN"
    records.tofile(path)
    result = io.read_points(path)
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result, records[:, :3])
    path.write_bytes(b"")
    assert io.read_points(path).shape == (0, 3)
    np.array([1, 2, 3], dtype=np.float32).tofile(path)
    with pytest.raises(ValueError, match="XYZI"):
        io.read_points(path)
    path.write_bytes(records.tobytes() + b"x")
    with pytest.raises(ValueError, match="XYZI"):
        io.read_points(path)


def test_float32_roundtrip_overwrite_and_path_spaces(tmp_path):
    rng = np.random.default_rng(17)
    bit_patterns = rng.integers(0, 2**32, size=3000, dtype=np.uint32)
    values = bit_patterns.view(np.float32)
    values = values[np.isfinite(values)]
    values = values[: (len(values) // 3) * 3].reshape(-1, 3)
    values = np.concatenate((values, np.array([[0.0, -0.0, np.nextafter(np.float32(0), np.float32(1))]], dtype=np.float32)))
    path = tmp_path / "roundtrip with spaces.ply"
    io.save_ply_ascii_geo(np.ones((10, 3)), path)
    io.save_ply_ascii_geo(values.astype(np.float64), path)
    result = io.read_points(path).astype(np.float32)
    np.testing.assert_array_equal(result.view(np.uint32), values.view(np.uint32))
    assert path.read_text().count("end_header") == 1
    assert b"\r\n" not in path.read_bytes()


def test_write_empty_and_invalid_shape(tmp_path):
    path = tmp_path / "empty.ply"
    io.save_ply_ascii_geo(np.empty((0, 3)), path)
    assert io.read_points(path).shape == (0, 3)
    with pytest.raises(ValueError, match="shape"):
        io.save_ply_ascii_geo(np.ones((3, 4)), path)


def test_read_many_preserves_order(tmp_path):
    paths = []
    for i in [4, 1, 3, 2]:
        path = tmp_path / f"cloud {i}.ply"
        io.save_ply_ascii_geo(np.full((i, 3), i), path)
        paths.append(path)
    for workers in (None, 1, 2, 20):
        clouds = io.read_point_clouds(iter(paths), workers=workers)
        assert [len(cloud) for cloud in clouds] == [4, 1, 3, 2]
        for cloud in clouds:
            np.testing.assert_array_equal(cloud, np.full_like(cloud, len(cloud)))
    assert io.read_point_clouds([]) == []
    with pytest.raises(ValueError, match="workers"):
        io.read_point_clouds(paths, workers=0)


def test_read_many_propagates_file_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        io.read_point_clouds([tmp_path / "missing.ply", tmp_path / "missing2.ply"])
