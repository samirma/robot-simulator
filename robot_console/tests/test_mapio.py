"""Map files: the map_server pgm/yaml pair, and the log-odds sidecar."""

from __future__ import annotations

import numpy as np
import pytest
from synthetic import mapped_room

from robot_console.slam import mapio
from robot_console.slam.grid import OCCUPIED, UNKNOWN, OccupancyGrid


def occupied_world(grid) -> list:
    """Where the obstacles are, in metres.

    The unit these tests care about. Saving trims the grid's unused margin, so array
    shape and index are not preserved across a round trip -- but the position of every
    wall on the floor must be, or a pose computed against the map is wrong.
    """
    cells = np.argwhere(grid.classify() == OCCUPIED)
    return sorted(map(tuple, np.round(grid.cell_to_world(cells[:, ::-1]), 6)))


def test_save_writes_the_three_files(tmp_path):
    grid = mapped_room()
    yaml_path = mapio.save_map(grid, tmp_path)
    assert yaml_path == tmp_path / "map.yaml"
    assert (tmp_path / "map.pgm").exists()
    assert (tmp_path / "map.npz").exists()


def test_the_yaml_is_what_map_server_expects(tmp_path):
    """navigation_active.launch feeds this straight to map_server; the keys are its API."""
    grid = mapped_room()
    meta = mapio.read_yaml(mapio.save_map(grid, tmp_path))
    assert meta["image"] == "map.pgm"
    assert meta["resolution"] == pytest.approx(grid.resolution)
    # The origin is the saved map's, which trimming moves; load_map must agree with it.
    assert meta["origin"][:2] == pytest.approx(list(mapio.load_map(tmp_path).origin))
    assert meta["origin"][2] == 0.0
    assert meta["negate"] == 0
    assert meta["occupied_thresh"] == pytest.approx(0.65)
    assert meta["free_thresh"] == pytest.approx(0.196)


def test_the_pgm_uses_map_servers_palette(tmp_path):
    import cv2

    grid = mapped_room()
    mapio.save_map(grid, tmp_path)
    img = cv2.imread(str(tmp_path / "map.pgm"), cv2.IMREAD_GRAYSCALE)
    assert set(np.unique(img)) <= {0, 205, 254}
    assert img.shape == mapio.load_map(tmp_path).shape


def test_the_pgm_is_flipped_so_row_zero_is_the_top(tmp_path):
    """A pgm's first row is the highest y; the grid's row 0 is the lowest."""
    grid = OccupancyGrid(0.1, width=4, height=4, origin=(0.0, 0.0))
    grid.data[0, 0] = 10.0   # occupied, at the world origin corner
    image = mapio.to_pgm_image(grid)
    assert image[-1, 0] == mapio.PGM_OCCUPIED
    assert image[0, 0] == mapio.PGM_UNKNOWN


def test_round_trip_through_the_sidecar_is_exact(tmp_path):
    grid = mapped_room()
    mapio.save_map(grid, tmp_path)
    back = mapio.load_map(tmp_path)
    assert back.resolution == grid.resolution
    assert occupied_world(back) == occupied_world(grid)
    # Exact log-odds, not just the three grey levels -- that is what the sidecar is for.
    assert set(np.unique(back.data).tolist()) == set(np.unique(grid.data).tolist())


def test_loading_without_the_sidecar_still_works(tmp_path):
    """A map produced on the robot has no npz; it must still load and be navigable."""
    grid = mapped_room()
    mapio.save_map(grid, tmp_path, sidecar=False)
    assert not (tmp_path / "map.npz").exists()
    back = mapio.load_map(tmp_path)
    assert occupied_world(back) == occupied_world(grid)
    assert (back.classify() == UNKNOWN).any(), "unknown space survives the pgm round trip"


def test_a_stale_sidecar_is_ignored(tmp_path):
    """The yaml is the authority on geometry; a mismatched npz is not to be trusted."""
    grid = mapped_room()
    mapio.save_map(grid, tmp_path)
    np.savez_compressed(
        tmp_path / "map.npz",
        data=np.zeros((3, 3), dtype=np.float32),
        resolution=np.float64(0.99),
        origin=np.array([99.0, 99.0]),
    )
    back = mapio.load_map(tmp_path)
    assert occupied_world(back) == occupied_world(grid)


@pytest.mark.parametrize("target", ["map.yaml", "map", ""])
def test_a_map_can_be_named_several_ways(tmp_path, target):
    grid = mapped_room()
    mapio.save_map(grid, tmp_path)
    loaded = mapio.load_map(tmp_path / target if target else tmp_path)
    assert occupied_world(loaded) == occupied_world(grid)


def test_loading_a_missing_map_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        mapio.load_map(tmp_path / "nothing")


def test_describe_summarises_a_map(tmp_path):
    mapio.save_map(mapped_room(), tmp_path)
    text = mapio.describe(tmp_path)
    assert "m^2 mapped" in text and "@ 0.050 m" in text


def test_describe_returns_none_for_a_missing_map(tmp_path):
    assert mapio.describe(tmp_path / "nope") is None


@pytest.mark.parametrize(
    "text, expected",
    [
        ("image: map.pgm\nresolution: 0.05\norigin: [1.0, -2.0, 0.0]\n",
         {"image": "map.pgm", "resolution": 0.05, "origin": [1.0, -2.0, 0.0]}),
        ('image: "a b.pgm"\nresolution: 0.1\norigin: [0, 0, 0]\nnegate: 1\n',
         {"image": "a b.pgm", "negate": 1}),
        ("image: m.pgm  # trailing\nresolution: 0.05\norigin: [0, 0, 0]\n",
         {"image": "m.pgm"}),
    ],
)
def test_the_yaml_reader_handles_what_map_server_writes(tmp_path, text, expected):
    path = tmp_path / "map.yaml"
    path.write_text(text)
    meta = mapio.read_yaml(path)
    for key, value in expected.items():
        assert meta[key] == value


def test_the_yaml_reader_refuses_what_it_cannot_parse(tmp_path):
    path = tmp_path / "map.yaml"
    path.write_text("image: m.pgm\nresolution: 0.05\norigin: [0,0,0]\nnested:\n  a: 1\n")
    with pytest.raises(ValueError):
        mapio.read_yaml(path)


def test_a_yaml_missing_a_required_key_raises(tmp_path):
    path = tmp_path / "map.yaml"
    path.write_text("image: m.pgm\n")
    with pytest.raises(ValueError):
        mapio.read_yaml(path)


def test_saving_creates_the_directory(tmp_path):
    target = tmp_path / "runs" / "house"
    mapio.save_map(mapped_room(), target)
    assert (target / "map.yaml").exists()


def test_a_saved_map_is_trimmed_to_its_content(tmp_path):
    """The grid grows in chunks and never shrinks; the file should not carry the slack."""
    grid = mapped_room()
    padded = OccupancyGrid(
        grid.resolution, origin=grid.origin - 5.0,
        data=np.zeros((grid.height + 200, grid.width + 200), dtype=np.float32),
    )
    padded.data[100:100 + grid.height, 100:100 + grid.width] = grid.data

    mapio.save_map(padded, tmp_path)
    saved = mapio.load_map(tmp_path)
    assert saved.height < padded.height and saved.width < padded.width


def test_trimming_does_not_move_the_map_in_the_world(tmp_path):
    """Cropping shifts cell (0, 0), so `origin` has to shift with it -- otherwise the
    whole map slides across the floor and every pose computed against it is wrong."""
    grid = mapped_room()
    mapio.save_map(grid, tmp_path)
    assert occupied_world(mapio.load_map(tmp_path)) == occupied_world(grid)


def test_trimming_keeps_a_border_of_context():
    trimmed = mapio.crop_to_content(mapped_room(), border_cells=4)
    known = np.argwhere(trimmed.data != 0.0)
    assert known[:, 0].min() >= 1 and known[:, 1].min() >= 1, "content is not flush to the edge"


def test_trimming_an_empty_map_is_harmless():
    empty = OccupancyGrid(0.05, width=32, height=32)
    assert mapio.crop_to_content(empty).shape == empty.shape


def test_trimming_can_be_switched_off(tmp_path):
    grid = mapped_room()
    mapio.save_map(grid, tmp_path, crop=False)
    assert mapio.load_map(tmp_path).shape == grid.shape
