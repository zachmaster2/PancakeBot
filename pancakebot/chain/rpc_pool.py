"""Pick the first RPC URL that responds with the expected chain id."""
from __future__ import annotations

import requests

from pancakebot.errors import InvariantError, TransientRpcError


def _eth_chain_id(rpc_url: str, *, timeout_seconds: int) -> int:
    payload = {"jsonrpc": "2.0", "id": 1, "method": "eth_chainId", "params": []}
    try:
        resp = requests.post(rpc_url, json=payload, timeout=timeout_seconds)
    except requests.RequestException as e:
        raise TransientRpcError(f"rpc_http_error: {e}") from e

    if resp.status_code != 200:
        raise TransientRpcError(f"rpc_http_status: {resp.status_code}")

    try:
        obj = resp.json()
    except ValueError as e:
        raise TransientRpcError(f"rpc_bad_json: {e}") from e

    if isinstance(obj, dict) and obj.get("error"):
        raise TransientRpcError(f"rpc_error: {obj['error']}")

    if not isinstance(obj, dict):
        raise TransientRpcError("rpc_response_not_dict")

    raw = obj.get("result")
    if not isinstance(raw, str):
        raise TransientRpcError("rpc_missing_result")

    try:
        return int(raw, 16)
    except ValueError as e:
        raise TransientRpcError(f"rpc_chain_id_not_hex: {raw}") from e


def choose_rpc_url(urls: list[str], *, expected_chain_id: int, timeout_seconds: int) -> str:
    """Pick the first healthy RPC URL that reports the expected chain id.

    Transient RPC errors are handled here by trying the next URL.
    If all URLs fail, raise a TransientRpcError("all_rpcs_down").
    """
    if not urls:
        raise InvariantError("rpc_urls_empty")

    for url in urls:
        try:
            cid = _eth_chain_id(url, timeout_seconds=timeout_seconds)
        except TransientRpcError:
            continue

        if cid != expected_chain_id:
            continue

        return url

    raise TransientRpcError("all_rpcs_down")
