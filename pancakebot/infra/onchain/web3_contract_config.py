from __future__ import annotations

from dataclasses import dataclass

from pancakebot.core.errors import InvariantError


@dataclass(frozen=True, slots=True)
class Web3ContractConfig:
    """Minimal on-chain contract configuration.

    Required fields:
      - rpc_url: selected by RpcPool (failover list is hardcoded elsewhere)
      - abi_json_path: path to ABI JSON file (must be a JSON list)
      - private_key: wallet private key (from env)
    """

    rpc_url: str
    abi_json_path: str
    private_key: str
    rpc_urls: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.rpc_url:
            raise InvariantError('rpc_url_required')
        if not self.abi_json_path:
            raise InvariantError('abi_json_path_required')
        if not self.private_key:
            raise InvariantError('private_key_required')
