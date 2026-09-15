"""FastAPI WebSocket server wrapping the flwcnx pipeline.

The app lives at `flwcnx.api.server:app`. It is not re-exported here: importing
the server from the package `__init__` makes `python -m flwcnx.api.server` load
the module twice, once as `flwcnx.api.server` and once as `__main__`.
"""
