import json
import tempfile
import unittest
from pathlib import Path

from utils.training_logger import append_epoch_metrics, write_training_summary


class TrainingLoggerTest(unittest.TestCase):
    def test_appends_epoch_metrics_and_writes_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            metrics_file = out / "metrics" / "epoch_metrics.csv"
            append_epoch_metrics(metrics_file, {"epoch": 1, "train_loss": 1.2, "val_IoU": 0.3})
            append_epoch_metrics(metrics_file, {"epoch": 2, "train_loss": 1.0, "val_IoU": 0.4})

            lines = metrics_file.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(lines[0], "epoch,train_loss,val_IoU")
            self.assertEqual(len(lines), 3)

            summary_file = out / "logs" / "training_summary.json"
            write_training_summary(summary_file, {"best_epoch": 2, "best_IoU": 0.4})
            self.assertEqual(json.loads(summary_file.read_text(encoding="utf-8"))["best_epoch"], 2)


if __name__ == "__main__":
    unittest.main()
