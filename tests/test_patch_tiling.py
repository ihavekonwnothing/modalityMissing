import unittest

from datasets.s1s2_water import _tile_grid


class PatchTilingTest(unittest.TestCase):
    def test_tile_grid_covers_full_scene_with_edge_tiles(self):
        tiles = _tile_grid(height=1000, width=1000, patch_size=512, stride=512)

        self.assertIn((0, 0), tiles)
        self.assertIn((488, 488), tiles)
        self.assertEqual(len(tiles), 4)

    def test_tile_grid_uses_patch_size_as_default_stride(self):
        self.assertEqual(_tile_grid(1024, 1024, 512), [(0, 0), (0, 512), (512, 0), (512, 512)])


if __name__ == "__main__":
    unittest.main()
