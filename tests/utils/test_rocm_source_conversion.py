# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import subprocess
import sys
from pathlib import Path

import pytest

# Run PyTorch's real preprocessor; these checks do not need a HIP toolchain.


@pytest.mark.parametrize("mode", ["out_of_tree", "in_source", "cmake"])
def test_hipify_converts_quoted_headers(tmp_path, mode):
    project = tmp_path / "source"
    project.mkdir()
    header = project / "nested" / "runtime.h"
    header.parent.mkdir()
    original = "#include <cuda_runtime.h>\n"
    header.write_text(original)
    source = project / "kernel.cu"
    source.write_text('#include "nested/runtime.h"\n')
    output = project if mode == "in_source" else tmp_path / "build" / "csrc"
    output.parent.mkdir(exist_ok=True)
    source_arg = "csrc/kernel.cu" if mode == "cmake" else str(source)
    script = Path(__file__).resolve().parents[2] / "cmake" / "hipify.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "-p",
            str(project),
            "-o",
            str(output),
            source_arg,
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=output.parent if mode == "cmake" else None,
    )
    converted = Path(result.stdout.strip().splitlines()[-1])
    assert converted.is_relative_to(output)
    assert converted.exists()
    converted_headers = list(output.rglob("*.h"))
    assert any("hip/hip_runtime.h" in path.read_text() for path in converted_headers)
    if mode != "in_source":
        assert header.read_text() == original
        assert not list(project.rglob("*.hip"))


def test_hipify_unchanged_source_has_build_path(tmp_path):
    project = tmp_path / "source"
    project.mkdir()
    source = project / "kernel.cu"
    source.write_text("int unchanged = 1;\n")
    output = tmp_path / "output"
    script = Path(__file__).resolve().parents[2] / "cmake" / "hipify.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "-p",
            str(project),
            "-o",
            str(output),
            str(source),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    converted = Path(result.stdout.strip().splitlines()[-1])
    assert converted.suffix == ".hip"
    assert converted.is_relative_to(output)
    assert converted.read_text() == source.read_text()
