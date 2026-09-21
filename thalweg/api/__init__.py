"""FastAPI WebSocket server wrapping the thalweg pipeline.

The app lives at `thalweg.api.server:app`. It is not re-exported here: importing
the server from the package `__init__` makes `python -m thalweg.api.server` load
the module twice, once as `thalweg.api.server` and once as `__main__`.
"""
