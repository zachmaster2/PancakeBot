"""Exception types for invariant violations and transient RPC/Graph failures."""


class InvariantError(Exception):
    pass


class TransientRpcError(Exception):
    pass


class TransientGraphError(Exception):
    pass
