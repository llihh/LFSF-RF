from PIL import Image
import pytest

from lfsf.data import FlatFocalStackDataset


def write_rgb(path, value):
    Image.new("RGB", (12, 8), color=(value, value, value)).save(path)


def test_flat_stack_layout_groups_and_naturally_sorts(tmp_path):
    for stack_id in ("10", "2"):
        for plane in range(3):
            write_rgb(tmp_path / f"{stack_id}_{plane}.png", plane * 40)
    dataset = FlatFocalStackDataset(
        str(tmp_path), expected_frames=3, image_size=0, max_stack_size=24
    )
    assert [item["name"] for item in dataset] == ["2", "10"]
    assert dataset[0]["stack"].shape == (3, 3, 8, 12)


def test_flat_stack_layout_rejects_missing_plane(tmp_path):
    write_rgb(tmp_path / "sample_0.png", 0)
    write_rgb(tmp_path / "sample_2.png", 80)
    with pytest.raises(RuntimeError, match="non-contiguous"):
        FlatFocalStackDataset(
            str(tmp_path), expected_frames=3, image_size=0, max_stack_size=24
        )
