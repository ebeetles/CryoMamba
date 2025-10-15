"""
Test environment shims for compatibility across library versions.
Loaded automatically by Python when present in sys.path.
"""

# Patch websockets client to provide `receive()` alias for older/newer versions
try:
    import websockets  # type: ignore

    targets = []
    if hasattr(websockets, 'client') and hasattr(websockets.client, 'ClientConnection'):
        targets.append(websockets.client.ClientConnection)
    if hasattr(websockets, 'legacy') and hasattr(websockets.legacy, 'client') and hasattr(websockets.legacy.client, 'WebSocketClientProtocol'):
        targets.append(websockets.legacy.client.WebSocketClientProtocol)
    if hasattr(websockets, 'asyncio') and hasattr(websockets.asyncio, 'client') and hasattr(websockets.asyncio.client, 'ClientConnection'):
        targets.append(websockets.asyncio.client.ClientConnection)

    for cls in targets:
        try:
            if not hasattr(cls, 'receive') and hasattr(cls, 'recv'):
                setattr(cls, 'receive', cls.recv)
        except Exception:
            continue

    # Provide a connect wrapper to ensure objects used via `async with` expose `receive()`
    _orig_connect = websockets.connect

    class _ConnectWrapper:
        def __init__(self, inner):
            self._inner = inner  # the original Connect object

        def __await__(self):  # Delegate plain await semantics
            return self._inner.__await__()

        async def __aenter__(self):
            proto = await self._inner.__aenter__()
            try:
                if not hasattr(proto, 'receive') and hasattr(proto, 'recv'):
                    # Create a thin proxy that forwards attributes and exposes receive()
                    class _ProtoProxy:
                        def __init__(self, p):
                            self._p = p
                        async def receive(self):
                            return await self._p.recv()
                        def __getattr__(self, name):
                            return getattr(self._p, name)
                    return _ProtoProxy(proto)
            except Exception:
                pass
            return proto

        async def __aexit__(self, exc_type, exc, tb):
            return await self._inner.__aexit__(exc_type, exc, tb)

    def _connect_wrapper(*args, **kwargs):  # noqa: ANN001
        return _ConnectWrapper(_orig_connect(*args, **kwargs))

    websockets.connect = _connect_wrapper  # type: ignore[assignment]
except Exception:
    # Silently ignore if websockets not available
    pass


