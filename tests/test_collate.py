import unittest

import torch

from utils.collate import segmentation_collate


class SegmentationCollateTest(unittest.TestCase):
    def test_keeps_metadata_as_list_and_stacks_tensors(self):
        samples = [
            {
                "image": torch.zeros(6, 8, 8),
                "sar": torch.zeros(2, 8, 8),
                "opt": torch.zeros(4, 8, 8),
                "mask": torch.zeros(1, 8, 8),
                "valid_mask": torch.ones(1, 8, 8),
                "sample_id": "a",
                "metadata": {"scene_metadata": {"s1_srcids": ["one"]}},
            },
            {
                "image": torch.ones(6, 8, 8),
                "sar": torch.ones(2, 8, 8),
                "opt": torch.ones(4, 8, 8),
                "mask": torch.ones(1, 8, 8),
                "valid_mask": torch.ones(1, 8, 8),
                "sample_id": "b",
                "metadata": {"scene_metadata": {"s1_srcids": ["one", "two"]}},
            },
        ]
        batch = segmentation_collate(samples)
        self.assertEqual(tuple(batch["image"].shape), (2, 6, 8, 8))
        self.assertEqual(batch["sample_id"], ["a", "b"])
        self.assertEqual(len(batch["metadata"]), 2)


if __name__ == "__main__":
    unittest.main()
