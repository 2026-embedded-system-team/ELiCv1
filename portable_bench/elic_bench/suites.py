"""Portable exactness and paired timing suites for ELiC's public primitives.

Every suite yields ``ctx.record_case(...)`` records.  Frozen reference code is
loaded by filename; optional Open3D/TorchSparse packages are never imported.
Only files below ``ctx.output_dir`` are created by these suites.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import itertools
import os
import sys
import tempfile
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import torch


def exact(before, after):
    """Require matching structure, tensor/array dtype, shape and all values."""
    if isinstance(before, torch.Tensor):
        assert isinstance(after, torch.Tensor)
        assert before.dtype == after.dtype and before.shape == after.shape
        assert torch.equal(before, after), "tensor values differ"
    elif isinstance(before, np.ndarray):
        assert isinstance(after, np.ndarray)
        assert before.dtype == after.dtype and before.shape == after.shape
        np.testing.assert_array_equal(before, after)
    elif isinstance(before, (tuple, list)):
        assert isinstance(after, type(before)) and len(before) == len(after)
        for old, new in zip(before, after):
            exact(old, new)
    else:
        assert before == after, "values differ"
    return {"exact": True}


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_modules(repo_root, device="cpu"):
    """Return isolated ``(reference, candidate)`` namespaces.

    The only reference adjustment prepopulates its formerly CUDA-only lookup
    tables on the actual test device.  Arithmetic and allocation code is intact.
    """
    root = Path(repo_root)
    token = hashlib.sha256(str(root.resolve()).encode()).hexdigest()[:12]
    refroot = root / "benchmarks" / "reference" / "src"
    reference = SimpleNamespace(**{
        name: _load(refroot / f"{name}.py", f"_elic_ref_{token}_{name}")
        for name in ("morton", "op")
    })
    pkgname = f"_elic_candidate_{token}"
    package = ModuleType(pkgname)
    package.__path__ = [str(root / "src")]
    sys.modules[pkgname] = package
    candidate = SimpleNamespace(**{
        name: _load(root / "src" / f"{name}.py", f"{pkgname}.{name}")
        for name in ("morton", "op", "io", "boe")
    })
    d = torch.device(device)
    reference.morton._get_occ_bits._occ_bits = (
        (torch.arange(256, device=d)[:, None] >>
         torch.arange(8, device=d)[None, :]) & 1).bool()
    reference.morton._get_expand_coords_base._table = torch.tensor(
        [[i & 1, (i >> 1) & 1, (i >> 2) & 1] for i in range(8)],
        dtype=torch.int32, device=d)
    return reference, candidate


def _reference_hierarchy(mt, coords, stop_bitdepth=2):
    if len(coords) == 0:
        return []
    points, codes = mt.morton3_sort(coords)
    result = []
    while True:
        points, codes, occ = mt.down_once(points, codes)
        result.append((points.clone(), occ.clone()))
        if int(points[:, 1:].max().item()).bit_length() <= stop_bitdepth:
            return result[::-1]


def _coords(ctx):
    rng = torch.Generator().manual_seed(ctx.config["seed"])
    coords = torch.randint(0, 16384, (ctx.config["points"], 4),
                           generator=rng, dtype=torch.int32)
    coords[:, 0] = 0
    # Include duplicates to exercise sorting's deduplication contract.
    if len(coords) > 8:
        coords[-8:] = coords[:8]
    return coords.to(ctx.device)


def _reference_histogram(op, coords, occ):
    if occ.numel() == 0:
        return torch.zeros(32, dtype=torch.float32, device=occ.device)
    _, values = op.sort_CF(coords, occ)
    low = torch.bincount(torch.remainder(values, 16).int().view(-1), minlength=16)
    high = torch.bincount(torch.div(values, 16, rounding_mode="floor").int().view(-1), minlength=16)
    counts = torch.cat((low, high)).float()
    return counts / counts.sum()


def _reference_collect(reference, coords, stop=7):
    hierarchy = _reference_hierarchy(reference.morton, coords, stop)
    if not hierarchy:
        return (torch.empty((0, 32), dtype=torch.float32, device=coords.device),
                torch.empty((0, 1), dtype=torch.int32, device=coords.device))
    values = torch.stack([_reference_histogram(reference.op, c, o) for c, o in hierarchy])
    labels = torch.arange(stop, stop + len(hierarchy), dtype=torch.int32,
                          device=coords.device).unsqueeze(1)
    return values, labels


def run_modules(ctx):
    """Morton, expansion/autograd, CDF, packing and offline BoE interfaces."""
    ref, new = load_modules(ctx.repo_root, ctx.device)
    coords = _coords(ctx)
    ordered, codes = ref.morton.morton3_sort(coords)
    parents, _, occ = ref.morton.down_once(ordered, codes)
    rng = torch.Generator().manual_seed(ctx.config["seed"])
    features = torch.randn((len(parents), ctx.config["channels"]), generator=rng).to(ctx.device)
    input_features = torch.randn((len(coords), ctx.config["channels"]), generator=rng).to(ctx.device)
    cases = [
        ("morton.code", lambda: ref.morton.morton3_code(coords[:, 1:]), lambda: new.morton.morton3_code(coords[:, 1:])),
        ("morton.sort", lambda: ref.morton.morton3_sort(coords), lambda: new.morton.morton3_sort(coords)),
        ("morton.down_once", lambda: ref.morton.down_once(ordered, codes), lambda: new.morton.down_once(ordered, codes)),
        ("morton.hierarchy", lambda: _reference_hierarchy(ref.morton, coords), lambda: new.morton.build_hierarchy(coords)),
        ("morton.upscale_coordinate", lambda: ref.morton.upscale_coordinate(parents, occ), lambda: new.morton.upscale_coordinate(parents, occ)),
        ("morton.upscale_feature", lambda: ref.morton.upscale_feature(occ, features), lambda: new.morton.upscale_feature(occ, features)),
        ("morton.upscale_coordinate_feature", lambda: ref.morton.upscale_coordinate_feature(parents, occ, features), lambda: new.morton.upscale_coordinate_feature(parents, occ, features)),
        ("boe.histogram", lambda: _reference_histogram(ref.op, parents, occ), lambda: new.boe.occupancy_histogram(occ)),
        ("boe.collect", lambda: _reference_collect(ref, coords), lambda: new.boe.collect_histograms(coords)),
        ("op.sort_C", lambda: ref.op.sort_C(coords), lambda: new.op.sort_C(coords)),
        ("op.sort_CF", lambda: ref.op.sort_CF(coords, input_features), lambda: new.op.sort_CF(coords, input_features)),
    ]
    for name, before, after in cases:
        yield ctx.record_case(name, "modules", before, after, check=exact,
                              details={"input_points": len(coords), "scope": "operator wall time", "reference_lookup_device": str(ctx.device), "uses_cuda": ctx.device.type == "cuda"})

    # Gradients have an independent count oracle; the frozen combined function
    # intentionally disables autograd and is not a valid gradient reference.
    def feature_grad():
        values = features.detach().clone().requires_grad_()
        _, expanded = new.morton.upscale_coordinate_feature(parents, occ, values)
        expanded.sum().backward()
        return values.grad

    expected_grad = torch.stack([((occ >> bit) & 1) for bit in range(8)]).sum(0).to(features.dtype)
    expected_grad = expected_grad[:, None].expand_as(features)
    yield ctx.record_case("morton.feature_gradient", "modules", lambda: expected_grad, feature_grad, check=exact,
                          details={"timing_scope": "correctness_only", "oracle": "occupied child count per parent/channel", "uses_cuda": ctx.device.type == "cuda"})

    for dtype in (torch.float16, torch.float32, torch.float64):
        for length in (2, 17, 257):
            cdf = torch.rand((min(ctx.config["points"], 4096), length), generator=rng).sort(-1).values.to(dtype=dtype, device=ctx.device)
            cdf[:, 0], cdf[:, -1] = 0, 1
            for normalized in (False, True):
                original = cdf.clone()
                def check_cdf(a, b, original=original, cdf=cdf):
                    exact(a, b)
                    exact(original, cdf)
                    return {"exact": True, "input_unchanged": True}
                yield ctx.record_case(f"cdf.{str(dtype).split('.')[-1]}.lp{length}.normalize{int(normalized)}", "modules",
                    lambda cdf=cdf, normalized=normalized: ref.op._convert_to_int_and_normalize(cdf, normalized),
                    lambda cdf=cdf, normalized=normalized: new.op._convert_to_int_and_normalize(cdf, normalized),
                    check=check_cdf, details={"cdf_rows": len(cdf), "cdf_columns": length, "input_dtype": str(dtype), "uses_cuda": ctx.device.type == "cuda"})
    boundary = torch.cat((torch.zeros((1, 8)), torch.ones((1, 9))), -1).to(ctx.device)
    halfway = ((torch.arange(65520, device=ctx.device, dtype=torch.float32) + .5) / 65520)[:, None].expand(-1, 17)
    for label, cdf in (("zero_probability", boundary), ("halfway_rounding", halfway)):
        yield ctx.record_case(f"cdf.{label}", "modules",
            lambda cdf=cdf: ref.op._convert_to_int_and_normalize(cdf, True),
            lambda cdf=cdf: new.op._convert_to_int_and_normalize(cdf, True), check=exact,
            details={"uses_cuda": ctx.device.type == "cuda"})

    # All packing formats are existing native-endian formats.  Compare raw
    # bytes as well as unpacked payloads; this is not a cross-endian rewrite.
    streams = [b"", b"abc\0\xff", bytes(range(256)), b"xyz" * 100]
    indices = [0, 255]
    def packing(module):
        a = module.pack_byte_stream_ls(streams)
        b = module.pack_byte_stream_and_enc_idx_list(streams, indices)
        c = module.pack_bitstream_and_enc_idx_list(b"".join(streams), indices)
        payload, ids = module.unpack_bitstream_and_enc_idx_list(c)
        result = (a, module.unpack_byte_stream(a), b,
                  module.unpack_byte_stream_and_enc_idx_list(b), c, payload, ids)
        assert result[1] == streams and result[3] == (streams, indices)
        assert payload == b"".join(streams) and ids.tolist() == indices
        return result
    yield ctx.record_case("stream.pack_unpack_all", "modules", lambda: packing(ref.op), lambda: packing(new.op), check=exact,
                          details={"uses_cuda": False, "byte_order": sys.byteorder, "formats": ["byte_stream_ls", "byte_stream_and_enc_idx_list", "bitstream_and_enc_idx_list"]})

    def edges():
        for size in (0, 1, 8):
            sample = torch.zeros((size, 4), dtype=torch.int32, device=ctx.device)
            exact(_reference_hierarchy(ref.morton, sample), new.morton.build_hierarchy(sample))
            exact(_reference_collect(ref, sample), new.boe.collect_histograms(sample))
        # Independent Python bit-interleaving oracle includes the 21-bit limit.
        xyz = [(0, 0, 0), (1, 2, 3), ((1 << 21) - 1, 0, 0), ((1 << 21) - 1,) * 3]
        oracle = [sum(((axis >> bit) & 1) << (3 * bit + k)
                      for bit in range(21) for k, axis in enumerate(point)) for point in xyz]
        actual = new.morton.morton3_code(torch.tensor(xyz, dtype=torch.int64, device=ctx.device))
        assert actual.tolist() == oracle
        # A batch-major oracle does not rely on the single-batch legacy code.
        rows = [(7, 1, 0, 0), (-3, 3, 2, 1), (7, 0, 0, 0), (-3, 1, 0, 0), (7, 1, 0, 0)]
        def interleave(row):
            return sum(((axis >> bit) & 1) << (3 * bit + k)
                       for bit in range(21) for k, axis in enumerate(row[1:]))
        expected_rows = sorted(set(rows), key=lambda row: (row[0], interleave(row)))
        multi = torch.tensor(rows, dtype=torch.int64, device=ctx.device)
        sorted_rows, sorted_codes = new.morton.morton3_sort(multi, batch_mode="multiple")
        assert sorted_rows.tolist() == [list(row) for row in expected_rows]
        assert sorted_codes.tolist() == [interleave(row) for row in expected_rows]
        parent, _, occupancy = new.morton.down_once(sorted_rows, sorted_codes)
        assert new.morton.upscale_coordinate(parent, occupancy).tolist() == sorted_rows.tolist()
        return True
    yield ctx.record_case("morton.boe_edge_cases", "modules", lambda: True, edges, check=exact,
                          details={"timing_scope": "correctness_only", "scope": "empty, shallow, duplicate coordinates and 21-bit boundary", "uses_cuda": ctx.device.type == "cuda"})


def _reference_io(root):
    path = Path(root) / "benchmarks" / "reference" / "src" / "io.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name in ("read_points", "save_ply_ascii_geo")]
    namespace = {"np": np, "os": os, "o3d": SimpleNamespace(
        geometry=SimpleNamespace(PointCloud=SimpleNamespace),
        utility=SimpleNamespace(Vector3dVector=lambda x: x))}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    return SimpleNamespace(**{name: namespace[name] for name in ("read_points", "save_ply_ascii_geo")})


def _oracle_read_points(path):
    """Independently parse the supported XYZ formats, without candidate code.

    Python tokenization and ``float`` conversion intentionally avoid the
    candidate PLY parser and its NumPy loadtxt implementation.
    """
    path = Path(path)
    if path.suffix.lower() == ".bin":
        if path.stat().st_size % 16:
            raise ValueError("Binary point cloud must contain float32 XYZI records")
        return np.fromfile(path, dtype=np.float32).reshape(-1, 4)[:, :3]
    with path.open("r", encoding="utf-8") as stream:
        if stream.readline().strip() != "ply":
            raise ValueError("Expected PLY signature")
        elements, current, ascii_format = [], None, False
        while True:
            line = stream.readline()
            if not line:
                raise ValueError("Missing PLY end_header")
            fields = line.split()
            if not fields or fields[0] in ("comment", "obj_info"):
                continue
            if fields[0] == "end_header":
                if len(fields) != 1:
                    raise ValueError("Malformed PLY end_header")
                break
            if fields[0] == "format":
                if fields != ["format", "ascii", "1.0"] or ascii_format:
                    raise ValueError("Only PLY ascii 1.0 is supported")
                ascii_format = True
            elif fields[0] == "element":
                if len(fields) != 3:
                    raise ValueError("Malformed PLY element")
                count = int(fields[2])
                if count < 0:
                    raise ValueError("Negative PLY element count")
                current = [fields[1], count, []]
                elements.append(current)
            elif fields[0] == "property":
                if current is None:
                    raise ValueError("PLY property without element")
                if current[0] == "vertex":
                    if len(fields) != 3 or fields[1] == "list":
                        raise ValueError("PLY vertex properties must be scalar")
                    current[2].append(fields[2])
            else:
                raise ValueError("Unsupported PLY header declaration")
        vertices = [element for element in elements if element[0] == "vertex"]
        if not ascii_format or len(vertices) != 1:
            raise ValueError("Expected one ASCII PLY vertex element")
        _, count, properties = vertices[0]
        if any(properties.count(axis) != 1 for axis in ("x", "y", "z")):
            raise ValueError("Expected exactly one x, y and z vertex property")
        columns = [properties.index(axis) for axis in ("x", "y", "z")]
        for name, preceding_count, _ in elements:
            if name == "vertex":
                break
            for _ in range(preceding_count):
                if not stream.readline():
                    raise ValueError("Truncated PLY preceding element")
        points = np.empty((count, 3), dtype=np.float64)
        for index in range(count):
            fields = stream.readline().split()
            if len(fields) != len(properties):
                raise ValueError("PLY vertex row does not match its properties")
            points[index] = [float(fields[column]) for column in columns]
        return points


def run_io(ctx):
    """Read/write and ordered batch I/O, with generated portable fixtures."""
    ref = _reference_io(ctx.repo_root)
    _, modules = load_modules(ctx.repo_root, ctx.device)
    new = modules.io
    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="io_fixtures_", dir=ctx.output_dir) as temp:
        folder = Path(temp)
        rng = np.random.default_rng(ctx.config["seed"])
        points = rng.normal(0, 30, (ctx.config["points"], 3)).astype(np.float32)
        generated = folder / "generated points.ply"
        new.save_ply_ascii_geo(points, generated)
        binary = folder / "generated.bin"
        np.column_stack((points, np.ones(len(points), np.float32))).astype(np.float32).tofile(binary)
        paths = [Path(item["path"]) for item in ctx.config.get("files", [])]
        paths.extend((generated, binary))
        sequence = itertools.count()
        oracle_points, legacy_supported = [], []
        for index, path in enumerate(paths):
            label = path.name
            oracle = _oracle_read_points(path)
            try:
                exact(oracle, ref.read_points(path))
                supported = True
            except Exception:
                supported = False
            oracle_points.append(oracle)
            legacy_supported.append(supported)
            details = {"uses_cuda": False, "file": str(path) if path not in (generated, binary) else label,
                       "scope": "OS-cached read, parse and close; no fsync",
                       "baseline": "frozen reference reader" if supported else "independent file-format oracle; frozen reader unsupported"}
            if supported:
                before = lambda path=path: ref.read_points(path)
            else:
                before = lambda oracle=oracle: oracle
                details["timing_scope"] = "correctness_only"
            def check_read(a, b, oracle=oracle):
                exact(oracle, a)
                return exact(oracle, b)
            yield ctx.record_case(f"io.read.{index}.{label}", "io", before,
                lambda path=path: new.read_points(path), check=check_read, details=details)
        for workers in (1, None):
            details = {"uses_cuda": False, "files": len(paths), "workers": workers,
                       "baseline": "sequential reader; legacy Pool(64) omitted for portability"}
            if all(legacy_supported):
                before = lambda: [ref.read_points(path) for path in paths]
            else:
                before = lambda: oracle_points
                details.update(timing_scope="correctness_only",
                               baseline="independent file-format oracle; frozen reader unsupported for one or more files")
            yield ctx.record_case(f"io.batch_read.workers_{workers}", "io",
                before,
                lambda workers=workers: new.read_point_clouds(paths, workers=workers), check=exact,
                details=details)
        def writer(fn, label):
            target = folder / f"{label}_{next(sequence)}.ply"
            fn(points, target)
            return target
        def check_write(a, b):
            expected = points.view(np.uint32)
            for path in (a, b):
                recovered = new.read_points(path).astype(np.float32)
                np.testing.assert_array_equal(recovered.view(np.uint32), expected)
            return {"float32_bit_exact": True}
        yield ctx.record_case("io.write.ply", "io", lambda: writer(ref.save_ply_ascii_geo, "reference"),
            lambda: writer(new.save_ply_ascii_geo, "candidate"), check=check_write,
            details={"uses_cuda": False, "points": len(points), "baseline": "unchanged writer AST; unused Open3D conversion stubbed; fresh filenames", "scope": "open, format and close; no fsync"})

        def edge_checks():
            special = folder / "extra properties.ply"
            special.write_text("ply\nformat ascii 1.0\nelement vertex 2\nproperty uchar red\nproperty float z\nproperty double x\nproperty float y\nelement face 1\nproperty list uchar int vertex_indices\nend_header\n255\t3 1 2\n10 6 4 5\n3 0 1 0\n", encoding="utf-8")
            np.testing.assert_array_equal(new.read_points(special), [[1, 2, 3], [4, 5, 6]])
            empty = folder / "empty.ply"
            new.save_ply_ascii_geo(np.empty((0, 3)), empty)
            assert new.read_points(empty).shape == (0, 3)
            overwrite = folder / "overwrite with spaces.ply"
            new.save_ply_ascii_geo(points, overwrite)
            new.save_ply_ascii_geo(points[:1], overwrite)
            np.testing.assert_array_equal(new.read_points(overwrite).astype(np.float32), points[:1])
            malformed = folder / "malformed.bin"
            malformed.write_bytes(b"bad")
            try:
                new.read_points(malformed)
            except ValueError:
                pass
            else:
                raise AssertionError("malformed XYZI binary was accepted")
            return True
        yield ctx.record_case("io.edge_cases", "io", lambda: True, edge_checks, check=exact,
            details={"uses_cuda": False, "timing_scope": "correctness_only", "scope": "named XYZ properties, faces excluded, empty cloud, overwrite, spaces and malformed binary"})


def geometry_inputs(ctx, io_module=None):
    """Yield ``(label, bxyz, metadata)`` with explicit quantization units.

    Match ``compress.py``: preserve the reader's PLY float64 / BIN float32
    dtype. Prequantized coordinates shift by offset only when negative.
    Otherwise divide by scale and add offset in NumPy before creating the
    tensor. Divide by posQ and round in PyTorch (ties to even).
    """
    if io_module is None:
        _, modules = load_modules(ctx.repo_root, ctx.device)
        io_module = modules.io
    files = ctx.config.get("files", [])
    if files:
        sources = []
        for index, item in enumerate(files):
            path = Path(item["path"])
            data = io_module.read_points(path)
            if not np.isfinite(data).all():
                raise ValueError(f"Nonfinite coordinates: {path}")
            sources.append((f"{index:03d}.{path.name}", data, bool(item["prequantized"]),
                            float(item.get("scale", .001)), float(item.get("offset", 131072)),
                            {"file": str(path), "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}))
    else:
        sources = [("synthetic", _coords(ctx)[:, 1:].cpu().numpy(), True, .001, 131072, {"synthetic": True})]
    for label, points, prequantized, scale, offset, meta in sources:
        for posq in ctx.config["posq"]:
            if prequantized:
                xyz = torch.tensor(points)
                if xyz.numel() and xyz.min() < 0:
                    xyz += offset
            else:
                if not np.isfinite(scale) or scale <= 0:
                    raise ValueError("Input scale must be finite and positive")
                xyz = torch.tensor(points / scale + offset)
            xyz = torch.round(xyz / posq)
            if xyz.numel() == 0:
                raise ValueError(f"Geometry requires a nonempty point cloud: {label}")
            if xyz.min() < 0 or xyz.max() >= (1 << 21):
                raise ValueError(f"Quantized geometry outside [0, 2**21): {label}, posQ={posq}")
            coords = torch.cat((torch.zeros((len(xyz), 1), dtype=torch.int32), xyz.int()), dim=1).to(ctx.device)
            details = dict(meta, posq=posq, input_points=len(coords), prequantized=prequantized,
                           scale=scale, offset=offset,
                           input_dtype=str(points.dtype),
                           quantization="preserve reader dtype (PLY float64 / BIN float32); prequantized tensor shifts +offset only if min<0; otherwise NumPy points/scale+offset before tensor conversion; PyTorch round / posQ",
                           quantized_sha256=hashlib.sha256(coords.cpu().numpy().tobytes()).hexdigest())
            yield f"{label}.q{posq}", coords, details


def cdf_for(coords, stage, op_module):
    """Non-neural, coordinate-conditioned 16-symbol int16 CDF on CPU."""
    cpu = coords.detach().to("cpu")
    values = torch.arange(1, 17, dtype=torch.float32)[None, :]
    shifts = ((cpu[:, 1:].sum(1) + stage) % 16).long()
    weights = values.expand(len(cpu), -1).gather(1, (torch.arange(16)[None, :] + shifts[:, None]) % 16)
    probs = weights / weights.sum(-1, keepdim=True)
    cdf = torch.cat((torch.zeros((len(cpu), 1)), probs.cumsum(-1)), -1)
    return np.ascontiguousarray(op_module._convert_to_int_and_normalize(cdf, True).numpy(), dtype=np.int16)


def run_geometry(ctx, reference_coder, candidate_coder):
    """Octree -> arithmetic stream -> coordinates, all four coder pairings.

    CPU arithmetic coding is timed alongside the selected tensor backend.
    Trained neural models and TorchSparse convolution are outside this suite.
    """
    ref, new = load_modules(ctx.repo_root, ctx.device)
    for label, coords, metadata in geometry_inputs(ctx, new.io):
        expected = ref.morton.morton3_sort(coords)[0]
        details = dict(metadata, uses_cuda=ctx.device.type == "cuda", cdf_model="deterministic coordinate-conditioned non-neural 16-symbol CDF",
                       scope=f"CPU arithmetic coder + {ctx.device} geometry tensors; includes CPU transfers/CDF; no trained model")

        def encode(mt, op, coder, legacy=False):
            hierarchy = _reference_hierarchy(mt, coords) if legacy else mt.build_hierarchy(coords)
            enc = coder.ArithmeticEncoder16()
            for parent, occ in hierarchy:
                for stage in (0, 1):
                    symbols = (occ & 15) if stage == 0 else (occ >> 4)
                    enc.encode_chunk(cdf_for(parent, stage, op), symbols.cpu().numpy().astype(np.int16))
            return enc.finish(), hierarchy[0][0]

        def check_encoded(a, b):
            exact(a, b)
            return {"bitstream_exact": True, "coarsest_coordinates_exact": True, "stream_bytes": len(a[0])}

        yield ctx.record_case(f"geometry.encode.{label}", "geometry",
            lambda: encode(ref.morton, ref.op, reference_coder, True),
            lambda: encode(new.morton, new.op, candidate_coder), check=check_encoded, details=details)
        # Generate streams outside decode timing, independently for each version.
        old_stream, old_coarse = encode(ref.morton, ref.op, reference_coder, True)
        new_stream, new_coarse = encode(new.morton, new.op, candidate_coder)
        level_count = len(new.morton.build_hierarchy(coords))

        def decode(stream, coarse, coder, mt, op):
            decoder = coder.ArithmeticDecoder16()
            decoder.reset(stream)
            parents = coarse.clone()
            for _ in range(level_count):
                low = decoder.decode_chunk(cdf_for(parents, 0, op)).astype(np.int32)
                high = decoder.decode_chunk(cdf_for(parents, 1, op)).astype(np.int32)
                occ = torch.from_numpy(low + (high << 4)).to(ctx.device)
                parents = mt.upscale_coordinate(parents, occ)
            return parents

        def check_decoded(a, b):
            exact(expected, a)
            exact(expected, b)
            cross_a = decode(new_stream, new_coarse, reference_coder, ref.morton, ref.op)
            cross_b = decode(old_stream, old_coarse, candidate_coder, new.morton, new.op)
            exact(expected, cross_a)
            exact(expected, cross_b)
            assert old_stream == new_stream
            return {"point_accuracy": 1.0, "exact_points": len(expected), "expected_points": len(expected),
                    "coder_combinations_passed": 4, "bitstream_exact": True}

        yield ctx.record_case(f"geometry.decode.{label}", "geometry",
            lambda: decode(old_stream, old_coarse, reference_coder, ref.morton, ref.op),
            lambda: decode(new_stream, new_coarse, candidate_coder, new.morton, new.op),
            check=check_decoded, details=dict(details, unique_points=len(expected), levels=level_count,
                                             stream_bytes=len(new_stream)))
