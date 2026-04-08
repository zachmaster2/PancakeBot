from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pancakebot.domain.types import Bet, Kline, Round
from pancakebot.infra.market_data_db import MarketDataDb, SqliteKlinesStore


def _round(epoch: int, *, with_bets: bool) -> Round:
    bets = ()
    if with_bets:
        bets = (
            Bet(wallet_address="0x1", amount_wei=1000, position="Bull", created_at=100),
            Bet(wallet_address="0x2", amount_wei=2000, position="Bear", created_at=101),
        )
    return Round(
        epoch=int(epoch),
        start_at=100 + int(epoch) * 10,
        lock_at=120 + int(epoch) * 10,
        close_at=180 + int(epoch) * 10,
        lock_price=600.0 + float(epoch),
        close_price=601.0 + float(epoch),
        position="Bull",
        failed=False,
        bets=bets,
    )


def _kline(open_ms: int, close_ms: int) -> Kline:
    return Kline(
        open_time_ms=int(open_ms),
        close_time_ms=int(close_ms),
        open_price=1.0,
        high_price=1.1,
        low_price=0.9,
        close_price=1.05,
        volume=10.0,
        quote_asset_volume=20.0,
    )


class MarketDataDbTests(unittest.TestCase):
    def test_rounds_and_klines_sync_and_read(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            rounds_path = root / "closed_rounds.jsonl"
            klines_path = root / "klines.jsonl"
            db_path = root / "market_data.sqlite"

            rounds = [_round(1, with_bets=True), _round(2, with_bets=False), _round(3, with_bets=True)]
            with rounds_path.open("w", encoding="utf-8") as f:
                for r in rounds:
                    f.write(json.dumps(r.to_json(), separators=(",", ":")) + "\n")

            klines = [
                _kline(0, 59_999),
                _kline(60_000, 119_999),
                _kline(120_000, 179_999),
                _kline(180_000, 239_999),
            ]
            with klines_path.open("w", encoding="utf-8") as f:
                for k in klines:
                    f.write(json.dumps(k.to_json(), separators=(",", ":")) + "\n")

            db = MarketDataDb(str(db_path))
            try:
                changed = db.ensure_sources_synced(
                    rounds_jsonl_path=str(rounds_path),
                    klines_jsonl_path=str(klines_path),
                )
                self.assertTrue(bool(changed["rounds_changed"]))
                self.assertTrue(bool(changed["klines_changed"]))
                self.assertEqual(3, int(db.count_rounds()))
                self.assertEqual(4, int(db.count_klines()))

                tail = db.load_tail_rounds(n=2)
                self.assertEqual([2, 3], [int(r.epoch) for r in tail])
                self.assertEqual(0, len(tail[0].bets))
                self.assertEqual(2, len(tail[1].bets))

                store = SqliteKlinesStore(market_data_db=db)
                self.assertEqual(0, int(store.earliest_open_time_ms()))
                self.assertEqual(180_000, int(store.latest_open_time_ms()))
                ctx = store.get_context_klines(anchor_close_time_ms=179_999, size=2)
                self.assertEqual([60_000, 120_000], [int(k.open_time_ms) for k in ctx])
            finally:
                db.close()


if __name__ == "__main__":
    unittest.main()
