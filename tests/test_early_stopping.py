import unittest

from train import EarlyStoppingState, build_early_stopping_config, update_early_stopping


class EarlyStoppingTest(unittest.TestCase):
    def test_disabled_never_requests_stop(self):
        cfg = build_early_stopping_config({"training": {"early_stopping": {"enabled": False, "patience": 1}}})
        state = EarlyStoppingState()

        for epoch in range(1, 5):
            state, improved, should_stop = update_early_stopping(state, 0.5, epoch, cfg)
            self.assertFalse(should_stop)

    def test_enabled_respects_patience_min_epochs_and_min_delta(self):
        cfg = build_early_stopping_config(
            {
                "training": {
                    "early_stopping": {
                        "enabled": True,
                        "patience": 2,
                        "min_epochs": 4,
                        "min_delta": 0.01,
                    }
                }
            }
        )
        state = EarlyStoppingState()

        state, improved, should_stop = update_early_stopping(state, 0.50, 1, cfg)
        self.assertTrue(improved)
        self.assertFalse(should_stop)
        state, improved, should_stop = update_early_stopping(state, 0.505, 2, cfg)
        self.assertFalse(improved)
        self.assertFalse(should_stop)
        state, improved, should_stop = update_early_stopping(state, 0.506, 3, cfg)
        self.assertFalse(improved)
        self.assertFalse(should_stop)
        state, improved, should_stop = update_early_stopping(state, 0.507, 4, cfg)
        self.assertFalse(improved)
        self.assertTrue(should_stop)


if __name__ == "__main__":
    unittest.main()
