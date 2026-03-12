from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pancakebot.infra.projection_cache_store import ProjectionCacheStore


class ProjectionCacheStoreTests(unittest.TestCase):
    def test_put_lookup_and_prune(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "projection_cache.sqlite"
            store = ProjectionCacheStore(str(path), commit_every_writes=1)
            try:
                hit, val = store.lookup_projection(
                    epoch=10,
                    lock_at=100,
                    cutoff_ts=90,
                    bull_wei=1000,
                    bear_wei=2000,
                )
                self.assertFalse(bool(hit))
                self.assertIsNone(val)

                store.put_projection(
                    epoch=10,
                    lock_at=100,
                    cutoff_ts=90,
                    bull_wei=1000,
                    bear_wei=2000,
                    projection=(3.0, 1.1, 1.9),
                )
                hit, val = store.lookup_projection(
                    epoch=10,
                    lock_at=100,
                    cutoff_ts=90,
                    bull_wei=1000,
                    bear_wei=2000,
                )
                self.assertTrue(bool(hit))
                self.assertIsNotNone(val)
                self.assertEqual((3.0, 1.1, 1.9), val)

                store.put_projection(
                    epoch=11,
                    lock_at=110,
                    cutoff_ts=100,
                    bull_wei=500,
                    bear_wei=700,
                    projection=None,
                )
                hit, val = store.lookup_projection(
                    epoch=11,
                    lock_at=110,
                    cutoff_ts=100,
                    bull_wei=500,
                    bear_wei=700,
                )
                self.assertTrue(bool(hit))
                self.assertIsNone(val)

                deleted = store.prune_before_or_equal_epoch(epoch=10)
                self.assertGreaterEqual(int(deleted), 1)
                hit, _ = store.lookup_projection(
                    epoch=10,
                    lock_at=100,
                    cutoff_ts=90,
                    bull_wei=1000,
                    bear_wei=2000,
                )
                self.assertFalse(bool(hit))
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
