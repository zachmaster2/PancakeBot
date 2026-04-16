"""The Graph client for PancakeSwap Prediction V2: fetches open, locked, and closed rounds with bets."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Literal

import requests

from pancakebot.types import Bet, Round
from pancakebot.util import InvariantError, TransientGraphError
from pancakebot.log import warn

RoundState = Literal["open", "locked", "closed"]


def _as_float_or_raise(value: Any, err_code: str) -> float:
    """Narrow a Graph API value (Any | None) to float or raise InvariantError."""
    if not isinstance(value, (int, float, str)):
        raise InvariantError(err_code)
    return float(value)


@dataclass(frozen=True, slots=True)
class GraphClient:
    """Thin The Graph client (fetch-only).

    Locked headers:
      - Content-Type: application/json
      - Authorization: Bearer {THE_GRAPH_API_KEY}

    Locked pagination rules:
      - closed rounds pages: rounds(first=1000, skip+=1000)
      - each round includes bets(first=1000)
      - only paginate a round's bets if the first page returned exactly 1000
    """

    endpoint: str
    api_key: str
    timeout_seconds: int = 30

    def __post_init__(self) -> None:
        if not self.endpoint:
            raise InvariantError("graph_endpoint_required")
        if not self.api_key:
            raise InvariantError("graph_api_key_required")
        if int(self.timeout_seconds) <= 0:
            raise InvariantError("graph_timeout_seconds_must_be_positive")

    # --------------------------
    # Public fetch APIs
    # --------------------------

    def fetch_latest_usable_closed_epoch(self) -> int:
        query = """query LatestUsableClosedEpoch {
  rounds(
    where: {
      failed: false,
      startAt_not: null,
      lockPrice_not: null,
      closePrice_not: null,
      position_in: [\"Bull\", \"Bear\", \"House\"]
    },
    first: 1,
    orderBy: epoch,
    orderDirection: desc
  ) { epoch }
}"""
        try:
            data = self._post(query, variables={})
        except TransientGraphError as e:
            warn("NET", "GRAPH", "FETCH", kind="closed_rounds_page", err=str(e))
            raise
        rounds = self._req_list(data.get("rounds"), "data.rounds")
        if len(rounds) != 1:
            raise InvariantError(f"latest_closed_epoch_bad_count: {len(rounds)}")
        epoch = self._req_obj(rounds[0], "round").get("epoch")
        return self._parse_int(epoch, "latest_closed_epoch.epoch")

    def fetch_latest_open_round(self) -> Round:
        """Fetch the latest open round (Graph)."""
        query_first = """query LatestOpenRoundFirst($first: Int!, $skip: Int!) {
  rounds(
    where: {
      startAt_not: null,
      lockPrice: null,
      closePrice: null,
      failed: null,
      position: null
    },
    first: 1,
    orderBy: epoch,
    orderDirection: desc
  ) {
    epoch startAt lockPrice closePrice position failed
    bets(first: $first, skip: $skip, orderBy: createdAt, orderDirection: asc) {
      amount position createdAt user { id }
    }
  }
}"""
        try:
            return self._fetch_latest_round_with_bets(state="open", query_first=query_first)
        except TransientGraphError as e:
            warn("NET", "GRAPH", "FETCH", kind="open_round", err=str(e))
            raise

    def fetch_latest_locked_round(self) -> Round:
        """Fetch the latest locked round (Graph)."""
        query_first = """query LatestLockedRoundFirst($first: Int!, $skip: Int!) {
  rounds(
    where: {
      startAt_not: null,
      lockPrice_not: null,
      closePrice: null,
      failed: null,
      position: null
    },
    first: 1,
    orderBy: epoch,
    orderDirection: desc
  ) {
    epoch startAt lockPrice closePrice position failed
    bets(first: $first, skip: $skip, orderBy: createdAt, orderDirection: asc) {
      amount position createdAt user { id }
    }
  }
}"""
        try:
            return self._fetch_latest_round_with_bets(state="locked", query_first=query_first)
        except TransientGraphError as e:
            warn("NET", "GRAPH", "FETCH", kind="locked_round", err=str(e))
            raise

    def fetch_open_round(self, epoch: int) -> Round:
        query_first = """query OpenRoundFirst($epoch: BigInt!, $first: Int!, $skip: Int!) {
  rounds(
    where: {
      epoch: $epoch,
      startAt_not: null,
      lockPrice: null,
      closePrice: null,
      failed: null,
      position: null
    },
    first: 1
  ) {
    epoch startAt lockPrice closePrice position failed
    bets(first: $first, skip: $skip, orderBy: createdAt, orderDirection: asc) {
      amount position createdAt user { id }
    }
  }
}"""
        return self._fetch_round_with_bets(epoch=epoch, state="open", query_first=query_first)

    def fetch_locked_round(self, epoch: int) -> Round:
        query_first = """query LockedRoundFirst($epoch: BigInt!, $first: Int!, $skip: Int!) {
  rounds(
    where: {
      epoch: $epoch,
      startAt_not: null,
      lockPrice_not: null,
      closePrice: null,
      failed: null,
      position: null
    },
    first: 1
  ) {
    epoch startAt lockPrice closePrice position failed
    bets(first: $first, skip: $skip, orderBy: createdAt, orderDirection: asc) {
      amount position createdAt user { id }
    }
  }
}"""
        return self._fetch_round_with_bets(epoch=epoch, state="locked", query_first=query_first)

    def fetch_closed_rounds(
        self,
        *,
        order: Literal["asc", "desc"],
        epoch_gte: int | None = None,
        epoch_lte: int | None = None,
        epoch_lt: int | None = None,
        first: int = 1000,
        skip: int = 0,
    ) -> list[Round]:
        """Fetch a *page* of usable closed rounds and fully materialize bets."""
        if order not in ("asc", "desc"):
            raise InvariantError("order_invalid")
        if epoch_gte is None and epoch_lte is None and epoch_lt is None:
            raise InvariantError("fetch_closed_rounds_requires_epoch_bound")

        where_parts = [
            "failed: false",
            "startAt_not: null",
            "lockPrice_not: null",
            "closePrice_not: null",
            'position_in: ["Bull","Bear","House"]',
        ]
        if epoch_gte is not None:
            where_parts.append("epoch_gte: $epoch_gte")
        if epoch_lte is not None:
            where_parts.append("epoch_lte: $epoch_lte")
        if epoch_lt is not None:
            where_parts.append("epoch_lt: $epoch_lt")

        vars_decl = []
        vars_obj: dict[str, Any] = {"first": int(first), "skip": int(skip)}
        if epoch_gte is not None:
            vars_decl.append("$epoch_gte: BigInt!")
            vars_obj["epoch_gte"] = str(int(epoch_gte))
        if epoch_lte is not None:
            vars_decl.append("$epoch_lte: BigInt!")
            vars_obj["epoch_lte"] = str(int(epoch_lte))
        if epoch_lt is not None:
            vars_decl.append("$epoch_lt: BigInt!")
            vars_obj["epoch_lt"] = str(int(epoch_lt))

        query = f"""query ClosedRoundsPage({', '.join(vars_decl)}, $first: Int!, $skip: Int!) {{
  rounds(
    where: {{{', '.join(where_parts)}}},
    first: $first,
    skip: $skip,
    orderBy: epoch,
    orderDirection: {order}
  ) {{
    epoch startAt lockPrice closePrice position failed
    bets(first: 1000, skip: 0, orderBy: createdAt, orderDirection: asc) {{
      amount position createdAt user {{ id }}
    }}
  }}
}}"""
        try:
            data = self._post(query, variables=vars_obj)
        except TransientGraphError as e:
            warn("NET", "GRAPH", "FETCH", kind="closed_rounds_page", err=str(e))
            raise

        rounds_payload = self._req_list(data.get("rounds"), "data.rounds")

        out: list[Round] = []
        for i, r in enumerate(rounds_payload):
            rr = self._req_obj(r, f"round[{i}]")
            bets0 = self._req_list(rr.get("bets"), f"round[{i}].bets")
            bets = self._parse_bets(bets0)

            if len(bets0) == 1000:
                epoch = self._parse_int(rr.get("epoch"), "round.epoch")
                bets = self._paginate_bets(epoch=epoch, existing=bets)

            out.append(self._parse_round(rr, state="closed", bets=bets))

        return out

    def fetch_additional_bets_in_round(self, *, epoch: int, skip: int) -> list[Bet]:
        """Fetch an additional bets page for a round (used when bets >= 1000)."""
        query = """query BetsPage($epoch: BigInt!, $first: Int!, $skip: Int!) {
  rounds(where: {epoch: $epoch}, first: 1) {
    bets(first: $first, skip: $skip, orderBy: createdAt, orderDirection: asc) {
      amount position createdAt user { id }
    }
  }
}"""
        first = 1000
        try:
            data = self._post(query, variables={"epoch": str(int(epoch)), "first": int(first), "skip": int(skip)})
        except TransientGraphError as e:
            warn("NET", "GRAPH", "FETCH", kind="round_bets_page", err=str(e))
            raise

        rounds = self._req_list(data.get("rounds"), "data.rounds")
        if len(rounds) != 1:
            raise InvariantError(f"bets_page_round_missing_or_ambiguous: epoch={epoch} count={len(rounds)}")

        r0 = self._req_obj(rounds[0], "round")
        bets_payload = self._req_list(r0.get("bets"), "round.bets")

        return self._parse_bets(bets_payload)

    # --------------------------
    # Internals
    # --------------------------

    def _post(self, query: str, *, variables: dict[str, Any]) -> dict[str, Any]:
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        payload = {"query": query, "variables": variables}

        try:
            resp = requests.post(self.endpoint, json=payload, headers=headers, timeout=int(self.timeout_seconds))
        except requests.RequestException as e:
            raise TransientGraphError(f"graph_http_error: {e}") from e

        if resp.status_code != 200:
            code = int(resp.status_code)
            body = resp.text[:500]
            # Invariant errors (developer/config): malformed request, auth/permission, wrong endpoint.
            if code in (400, 401, 403):
                raise InvariantError(f"graph_http_status_invariant: {code} body={body}")
            if code == 404:
                raise InvariantError(f"graph_http_status_not_found: {code} body={body}")

            # Transient errors: rate limit or Graph/backend failures.
            if code == 429 or (500 <= code <= 599):
                raise TransientGraphError(f"graph_http_status_transient: {code} body={body}")

            # Default: treat as invariant (unexpected permanent-ish response).
            raise InvariantError(f"graph_http_status_unexpected: {code} body={body}")

        try:
            obj = resp.json()
        except ValueError as e:
            # Can happen transiently (proxy/edge truncation) even with a 200.
            raise TransientGraphError(f"graph_bad_json: {e}") from e

        if isinstance(obj, dict) and obj.get("errors"):
            # GraphQL "errors" indicates a bad query/schema mismatch (developer/config).
            raise InvariantError(f"graph_errors_invariant: {obj['errors']}")

        if not isinstance(obj, dict) or "data" not in obj:
            raise InvariantError("graph_missing_data_invariant")

        data = obj["data"]
        if not isinstance(data, dict):
            raise InvariantError("graph_data_not_object_invariant")
        return data

    def _fetch_round_with_bets(self, *, epoch: int, state: RoundState, query_first: str) -> Round:
        first = 1000
        skip = 0
        data = self._post(query_first, variables={"epoch": str(int(epoch)), "first": int(first), "skip": int(skip)})

        rounds = self._req_list(data.get("rounds"), "data.rounds")
        if len(rounds) != 1:
            raise InvariantError(f"{state}_round_not_found_or_ambiguous: epoch={epoch} count={len(rounds)}")

        r0 = self._req_obj(rounds[0], "round")
        bets0 = self._req_list(r0.get("bets"), "round.bets")

        bets = self._parse_bets(bets0)
        if len(bets0) == first:
            bets = self._paginate_bets(epoch=epoch, existing=bets)

        return self._parse_round(r0, state=state, bets=bets)

    def _fetch_latest_round_with_bets(self, *, state: RoundState, query_first: str) -> Round:
        first = 1000
        skip = 0
        data = self._post(query_first, variables={"first": int(first), "skip": int(skip)})

        rounds = self._req_list(data.get("rounds"), "data.rounds")
        if len(rounds) != 1:
            raise InvariantError(f"latest_{state}_round_not_found_or_ambiguous: count={len(rounds)}")

        r0 = self._req_obj(rounds[0], "round")
        epoch = self._parse_int(r0.get("epoch"), "round.epoch")
        bets0 = self._req_list(r0.get("bets"), "round.bets")

        bets = self._parse_bets(bets0)
        if len(bets0) == first:
            bets = self._paginate_bets(epoch=epoch, existing=bets)

        return self._parse_round(r0, state=state, bets=bets)

    def _paginate_bets(self, *, epoch: int, existing: list[Bet]) -> list[Bet]:
        first = 1000
        skip = first
        bets = list(existing)

        while True:
            page = self.fetch_additional_bets_in_round(epoch=epoch, skip=skip)
            bets.extend(page)
            if len(page) < first:
                break
            skip += first

        return bets

    def _parse_round(self, r: dict[str, Any], *, state: RoundState, bets: list[Bet]) -> Round:
        from pancakebot.market_data.contract_constants import load_contract_constants
        interval_seconds = load_contract_constants().interval_seconds

        epoch = self._parse_int(r.get("epoch"), "round.epoch")
        start_at = self._parse_int(r.get("startAt"), "round.startAt")

        lock_price = r.get("lockPrice")
        close_price = r.get("closePrice")
        position = r.get("position")
        failed = r.get("failed")

        if state == "open":
            if any(x is not None for x in (lock_price, close_price, position, failed)):
                raise InvariantError("open_round_invariant_violation")
            return Round(
                epoch=epoch,
                start_at=start_at,
                lock_at=start_at + interval_seconds,
                lock_price=None,
                close_price=None,
                position=None,
                failed=None,
                bets=tuple(bets),
            )

        if state == "locked":
            lock_price_f = _as_float_or_raise(lock_price, "locked_round_missing_lock_price")
            if any(x is not None for x in (close_price, position, failed)):
                raise InvariantError("locked_round_invariant_violation")
            return Round(
                epoch=epoch,
                start_at=start_at,
                lock_at=start_at + interval_seconds,
                lock_price=lock_price_f,
                close_price=None,
                position=None,
                failed=None,
                bets=tuple(bets),
            )

        # closed usable
        if failed is not False:
            raise InvariantError("closed_round_failed_not_false")
        lock_price_closed = _as_float_or_raise(lock_price, "closed_round_missing_lockPrice")
        close_price_closed = _as_float_or_raise(close_price, "closed_round_missing_closePrice")
        if position is None:
            raise InvariantError("closed_round_missing_position")

        pos = str(position)
        if pos not in ("Bull", "Bear", "House"):
            raise InvariantError("closed_round_invalid_position")

        return Round(
            epoch=epoch,
            start_at=start_at,
            lock_at=start_at + interval_seconds,
            lock_price=lock_price_closed,
            close_price=close_price_closed,
            position=pos,
            failed=False,
            bets=tuple(bets),
        )

    def _parse_bets(self, bets_payload: list[Any]) -> list[Bet]:
        out: list[Bet] = []
        for i, b in enumerate(bets_payload):
            bb = self._req_obj(b, f"bet[{i}]")
            user = self._req_obj(bb.get("user"), f"bet[{i}].user")
            user_id = user.get("id")
            if user_id is None:
                raise InvariantError(f"bet[{i}].user.id_missing")

            amount_raw = bb.get("amount")
            if amount_raw is None:
                raise InvariantError(f"bet[{i}].amount_missing")

            try:
                dec = Decimal(str(amount_raw))
            except (InvalidOperation, TypeError) as e:
                raise InvariantError(f"bet[{i}].amount_bad_decimal: {e}") from e

            wei_dec = dec * Decimal("1000000000000000000")
            if wei_dec % 1 != 0:
                raise InvariantError(f"bet[{i}].amount_not_wei_exact")

            amount_wei = int(wei_dec)
            if amount_wei <= 0:
                raise InvariantError(f"bet[{i}].amount_wei_nonpositive")

            pos = bb.get("position")
            if pos not in ("Bull", "Bear"):
                raise InvariantError(f"bet[{i}].position_invalid")

            created_at = self._parse_int(bb.get("createdAt"), f"bet[{i}].createdAt")
            out.append(
                Bet(
                    wallet_address=str(user_id),
                    amount_wei=amount_wei,
                    position=pos,
                    created_at=created_at,
                )
            )
        return out

    @staticmethod
    def _req_obj(v: Any, ctx: str) -> dict[str, Any]:
        if not isinstance(v, dict):
            raise InvariantError(f"expected_object: {ctx}")
        return v

    @staticmethod
    def _req_list(v: Any, ctx: str) -> list[Any]:
        if not isinstance(v, list):
            raise InvariantError(f"expected_list: {ctx}")
        return v

    @staticmethod
    def _parse_int(v: Any, ctx: str) -> int:
        try:
            return int(v)
        except (TypeError, ValueError) as e:
            raise InvariantError(f"invalid_int: {ctx} err={e}") from e
