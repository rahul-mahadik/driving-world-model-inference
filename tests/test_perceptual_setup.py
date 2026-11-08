import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/prepare_perceptual_evaluation.py"
SPEC = importlib.util.spec_from_file_location("perceptual_setup", SCRIPT)
setup = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(setup)


class Response(io.BytesIO):
    def geturl(self):
        return setup.WHEEL_URL


def test_pinned_download_publishes_only_matching_bytes(tmp_path):
    payload = b"known package fixture"
    digest = hashlib.sha256(payload).hexdigest()
    target = tmp_path / "package.whl"
    result = setup.download_pinned(setup.WHEEL_URL, target, digest, len(payload), opener=lambda *a, **k: Response(payload))
    assert target.read_bytes() == payload
    assert result["sha256"] == digest
    assert not result["cache_hit"]
    again = setup.download_pinned(setup.WHEEL_URL, target, digest, len(payload), offline=True)
    assert again["cache_hit"]


def test_hash_mismatch_never_publishes_partial_file(tmp_path):
    payload = b"damaged"
    target = tmp_path / "package.whl"
    with pytest.raises(setup.PreparationError, match="sha256_mismatch"):
        setup.download_pinned(setup.WHEEL_URL, target, "0" * 64, len(payload), opener=lambda *a, **k: Response(payload))
    assert not target.exists()
    assert not list(tmp_path.glob("*.partial"))


def test_corrupt_cache_is_not_silently_replaced(tmp_path):
    target = tmp_path / "package.whl"
    target.write_bytes(b"corrupt")
    with pytest.raises(setup.PreparationError, match="sha256_mismatch"):
        setup.download_pinned(setup.WHEEL_URL, target, "0" * 64, 7, opener=lambda *a, **k: pytest.fail("network must not be used"))
    assert target.read_bytes() == b"corrupt"


def test_size_and_url_guards(tmp_path):
    with pytest.raises(setup.PreparationError, match="exceeds_pinned_size"):
        setup.download_pinned(setup.WHEEL_URL, tmp_path / "file", "0" * 64, 1, opener=lambda *a, **k: Response(b"too large"))
    with pytest.raises(setup.PreparationError, match="not_allowlisted"):
        setup.download_pinned("https://untrusted.invalid/file", tmp_path / "file", "0" * 64, 1)


def test_metadata_cannot_change_the_hard_pin(tmp_path):
    payload = {"info": {"version": "0.1.4"}, "urls": [{"filename": setup.WHEEL_NAME, "url": setup.WHEEL_URL, "size": setup.WHEEL_BYTES, "digests": {"sha256": "0" * 64}}]}
    (tmp_path / "lpips-0.1.4-pypi.json").write_text(json.dumps(payload))
    with pytest.raises(setup.PreparationError, match="pin_mismatch"):
        setup.verify_pypi_metadata(tmp_path, offline=True)


def test_target_cannot_be_the_inference_environment():
    with pytest.raises(setup.PreparationError, match="outside_current_environment"):
        setup.stage_wheel(Path("not-read.whl"), Path(sys.prefix) / "not-created")
