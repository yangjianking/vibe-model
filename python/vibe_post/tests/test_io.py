"""`vibe_post.io` 单元测试：三后端探测、`.vibebin` 读写、边界情形。

运行：
    pytest tests/test_io.py -q
    python tests/test_io.py            # 无 pytest 时的兜底

`.vibebin` 的布局与 C++ `vibe::io::BinaryWriter` **字节级一致**：
文件头 40 字节、每条记录 `name[32] + float64 数据`（行主序），
变量形状由名字唯一决定（见 `shape_for_name`）。

全部使用 numpy 合成数据，不依赖网络与外部文件。
文献：[D16] WRF ARW 第 5 章（后备二进制格式与 I/O API 设计）。
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from vibe_post.io import (
    VIBE_BIN_HEADER_BYTES,
    VIBE_BIN_MAGIC,
    Dataset,
    MissingDependencyError,
    UnsupportedFormatError,
    VibeBinError,
    VibeBinFile,
    VibeBinHeader,
    available_backends,
    normalize_dims,
    open_dataset,
    read_binary_header,
    read_vibebin,
    shape_for_name,
    sniff_format,
    stagger_from_name,
    summarize,
    to_numpy,
    write_vibebin,
)


def make_dataset(nx: int = 5, ny: int = 4, nz: int = 3, *,
                 real_kind: str = "float64") -> Dataset:
    """构造一个最小可用的合成 Dataset（全部体心变量）。"""
    rng = np.random.default_rng(7)
    shape = (nz, ny, nx)
    dtype = np.float32 if real_kind == "float32" else np.float64
    fields = {
        "theta": (300.0 + np.arange(nz, dtype=np.float64)[:, None, None] * 3.0
                  + rng.normal(scale=0.1, size=shape)).astype(dtype),
        "qv": np.full(shape, 0.008, dtype=dtype),
    }
    return Dataset(
        fields=fields,
        dims={name: ("zeta", "y", "x") for name in fields},
        coords={"zeta": np.linspace(0.0, 1000.0, nz),
                "x": (np.arange(nx) + 0.5) * 100.0,
                "y": (np.arange(ny) + 0.5) * 100.0},
        attrs={"nx": nx, "ny": ny, "nz": nz, "dx": 100.0, "dy": 100.0,
               "real_kind": real_kind},
        time=1800.0,
    )


# ---------------------------------------------------------------------------


def test_binary_layout_matches_cpp_spec(tmp_path: Path) -> None:
    """默认（classic）布局必须与 C++ 规范逐字节一致：40 + nvars*(32 + N*8)。"""
    dataset = make_dataset(nx=5, ny=4, nz=3)
    path = tmp_path / "classic.vibebin"
    write_vibebin(path, dataset)
    raw = path.read_bytes()

    assert raw[:8] == VIBE_BIN_MAGIC
    header = VibeBinHeader.unpack(raw)
    assert (header.nx, header.ny, header.nz) == (5, 4, 3)
    assert header.real_kind == 0
    assert header.nvars == 2
    assert header.nlevels == 3
    assert header.time == pytest.approx(1800.0)
    assert VIBE_BIN_HEADER_BYTES == 40
    # 每条记录：32 字节名字 + 5*4*3 个 float64（数据始终 float64）
    assert len(raw) == 40 + 2 * (32 + 5 * 4 * 3 * 8)
    # 第一条记录的名字与数据紧跟文件头
    assert raw[40:72].split(b"\x00", 1)[0] == b"theta"
    first = np.frombuffer(raw, dtype="<f8", count=60, offset=72)
    np.testing.assert_allclose(first.reshape(3, 4, 5), dataset.fields["theta"], rtol=0)

    assert read_binary_header(path).nvars == 2


def test_vibebin_roundtrip_float64(tmp_path: Path) -> None:
    """float64 往返：形状、数值、时间与属性都必须无损。"""
    dataset = make_dataset()
    path = tmp_path / "roundtrip.vibebin"
    write_vibebin(path, dataset)
    loaded = read_vibebin(path)
    assert set(loaded.fields) >= {"theta", "qv"}
    assert loaded.grid_shape() == (5, 4, 3)
    assert loaded.time == pytest.approx(1800.0)
    np.testing.assert_allclose(loaded.fields["theta"], dataset.fields["theta"], rtol=0, atol=0)
    np.testing.assert_allclose(loaded.fields["qv"], 0.008)
    assert loaded.attrs["nx"] == 5
    assert loaded.attrs["layout"] == "classic"


def test_extended_layout_roundtrip_with_attrs(tmp_path: Path) -> None:
    """extended=True 时带 CRC32 与 JSON 属性尾部；读取器必须能自动识别。"""
    dataset = make_dataset()
    path = tmp_path / "extended.vibebin"
    write_vibebin(path, dataset, extended=True, attrs={"run": "tutorial", "step": 12})
    loaded = read_vibebin(path)
    assert loaded.attrs["layout"] == "extended"
    assert loaded.attrs["run"] == "tutorial"
    assert loaded.attrs["step"] == 12
    np.testing.assert_allclose(loaded.fields["theta"], dataset.fields["theta"], rtol=0)

    # 损坏 CRC32 必须被检出
    raw = bytearray(path.read_bytes())
    crc_offset = 40 + 32 + 12          # 文件头 + 名字 + 形状
    struct.pack_into("<I", raw, crc_offset, 0xDEADBEEF)
    broken = tmp_path / "broken_crc.vibebin"
    broken.write_bytes(bytes(raw))
    with pytest.raises((VibeBinError, UnsupportedFormatError)):
        read_vibebin(broken)


def test_shape_rules_match_cpp(tmp_path: Path) -> None:
    """变量形状由名字决定（C++ shape_for_name 规则）。"""
    assert shape_for_name("u", 10, 8, 4) == (11, 8, 4)
    assert shape_for_name("v", 10, 8, 4) == (10, 9, 4)
    assert shape_for_name("w", 10, 8, 4) == (10, 8, 5)
    assert shape_for_name("zeta", 10, 8, 4) == (10, 8, 5)
    assert shape_for_name("theta", 10, 8, 4) == (10, 8, 4)
    assert shape_for_name("__restart_info__", 10, 8, 4) == (2, 1, 1)
    assert stagger_from_name("u") == "face_x"
    assert stagger_from_name("theta") == "cell"

    # 面场必须按名字隐含形状写入
    handle = VibeBinFile(mode="w")
    handle.set_grid(4, 3, 2)
    handle.add_variable("u", np.zeros((2, 3, 5)))       # (nz, ny, nx+1)
    handle.add_variable("theta", np.zeros((2, 3, 4)))
    path = tmp_path / "stagger.vibebin"
    handle.write(path)
    raw = path.read_bytes()
    assert len(raw) == 40 + 32 + 2 * 3 * 5 * 8 + 32 + 2 * 3 * 4 * 8
    loaded = read_vibebin(path)
    assert loaded.fields["u"].shape == (2, 3, 5)
    assert loaded.fields["theta"].shape == (2, 3, 4)

    with pytest.raises(VibeBinError):
        bad = VibeBinFile(mode="w")
        bad.set_grid(4, 3, 2)
        bad.add_variable("u", np.zeros((2, 3, 4)))      # 少了 x 面那一列
    with pytest.raises(VibeBinError):
        bad2 = VibeBinFile(mode="w")
        bad2.set_grid(4, 3, 2)
        bad2.add_variable("theta", np.zeros((2, 3)))


def test_vibebin_handles_nan_and_extremes(tmp_path: Path) -> None:
    """NaN / ±Inf / 全 NaN 场都能写入并原样读回（NaN 逐点比较）。"""
    dataset = make_dataset()
    field = dataset.fields["theta"].copy()
    field[0, 0, 0] = np.nan
    field[1, 1, 1] = np.inf
    field[2, 2, 2] = -np.inf
    dataset = dataset.with_field("theta", field)
    all_nan = np.full(dataset.fields["qv"].shape, np.nan)
    dataset = dataset.with_field("qv", all_nan)

    path = tmp_path / "nan.vibebin"
    write_vibebin(path, dataset)
    loaded = read_vibebin(path)
    np.testing.assert_array_equal(np.isnan(loaded.fields["theta"]), np.isnan(field))
    finite = np.isfinite(field)
    np.testing.assert_allclose(loaded.fields["theta"][finite], field[finite])
    assert np.all(np.isnan(loaded.fields["qv"]))

    handle = VibeBinFile(path)
    problems = handle.self_check()
    assert any("Inf" in problem for problem in problems)


def test_vibebin_corrupt_truncation_raises(tmp_path: Path) -> None:
    """截断 / 错误 magic 的文件必须给出明确的错误，而不是静默读出垃圾。"""
    dataset = make_dataset()
    path = tmp_path / "trunc.vibebin"
    write_vibebin(path, dataset)
    data = path.read_bytes()
    for cut in (4, 20, 39, 60, len(data) // 2):
        broken = tmp_path / "cut_{0}.vibebin".format(cut)
        broken.write_bytes(data[:cut])
        with pytest.raises((VibeBinError, ValueError)):
            read_vibebin(broken)

    bad_magic = tmp_path / "bad_magic.vibebin"
    bad_magic.write_bytes(b"NOTVIBE!" + data[8:])
    with pytest.raises(VibeBinError):
        read_vibebin(bad_magic)


def test_backend_detection_and_sniff(tmp_path: Path) -> None:
    """扩展名与文件头联合判定后端；未知格式在 auto 模式下必须报错。"""
    status = available_backends()
    assert status["vibebin"] is True
    assert set(status) == {"vibebin", "netcdf", "grib"}

    path = tmp_path / "file.vibebin"
    write_vibebin(path, make_dataset())
    assert sniff_format(path) == "vibebin"

    junk = tmp_path / "junk.vibebin"
    junk.write_bytes(b"\x00\x01\x02\x03" * 32)
    with pytest.raises(UnsupportedFormatError):
        open_dataset(junk)
    with pytest.raises((UnsupportedFormatError, MissingDependencyError)):
        open_dataset(junk, backend="netcdf")


def test_netcdf_backend_missing_dependency_or_roundtrip(tmp_path: Path) -> None:
    """NetCDF 后端：有 xarray 时往返一致，无依赖时抛带提示的 MissingDependencyError。"""
    from vibe_post.io import read_netcdf, to_netcdf

    dataset = make_dataset()
    path = tmp_path / "out.nc"
    try:
        to_netcdf(path, dataset)
    except MissingDependencyError as exc:
        assert "xarray" in str(exc) or "netcdf" in str(exc)
        return
    loaded = read_netcdf(path)
    np.testing.assert_allclose(loaded.fields["theta"], dataset.fields["theta"], rtol=1e-6)
    assert loaded.grid_shape() == dataset.grid_shape()


def test_dataset_helpers_and_summary(tmp_path: Path) -> None:
    """Dataset 的维度推断、select_time、merge、summarize 与 to_numpy。"""
    dataset = make_dataset()
    assert dataset.shape_of("theta") == (5, 4, 3)     # C++ 顺序 (nx, ny, nz)

    summed = summarize(dataset)
    assert "theta" in summed and "grid" in summed
    assert "nan=0" in summed

    stacked = Dataset(
        fields={"theta": np.stack([dataset.fields["theta"], dataset.fields["theta"]])},
        dims={"theta": ("time", "zeta", "y", "x")},
        coords={"time": np.array([0.0, 3600.0])},
    )
    first = stacked.select_time(0)
    assert first.fields["theta"].shape == (3, 4, 5)
    assert first.dims["theta"] == ("zeta", "y", "x")

    merged = dataset.merge(Dataset(fields={"extra": np.zeros((4, 5))}))
    assert "extra" in merged.fields
    assert normalize_dims(("time", "z", "lat", "lon")) == ("time", "zeta", "y", "x")
    np.testing.assert_array_equal(to_numpy(np.arange(3)), np.arange(3))
    with pytest.raises(TypeError):
        to_numpy(dataset)


# ---------------------------------------------------------------------------
# 无 pytest 时的兜底运行
# ---------------------------------------------------------------------------


def _run_all() -> int:
    import traceback
    import tempfile

    failures = 0
    for name, function in sorted(globals().items()):
        if not name.startswith("test_") or not callable(function):
            continue
        with tempfile.TemporaryDirectory() as tmp:
            try:
                function(Path(tmp))
                print("PASS", name)
            except Exception:
                failures += 1
                print("FAIL", name)
                traceback.print_exc()
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
