import os

import benchmark_full
from backend.core.engine import AICompressionEngine


def test_roundtrip_simple(tmp_path):
    input_file = tmp_path / "test.txt"
    output_file = tmp_path / "test.aicp"
    decompressed_file = tmp_path / "test_out.txt"

    content = b"Hello World! This is a test for AI compression roundtrip." * 10
    input_file.write_bytes(content)

    engine = AICompressionEngine()

    # Compress
    engine.compress(str(input_file), str(output_file))

    assert os.path.exists(output_file)
    assert os.path.getsize(output_file) > 0

    # Decompress
    engine.decompress(str(output_file), str(decompressed_file))

    assert os.path.exists(decompressed_file)
    assert decompressed_file.read_bytes() == content


def test_benchmark_runs():
    # Just ensure it doesn't crash. We use n=10 for speed in tests.
    benchmark_full.benchmark(n=10)
