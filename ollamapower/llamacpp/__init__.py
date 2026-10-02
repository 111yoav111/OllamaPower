"""llama.cpp: the pinned release, its download and verification, and the binaries we run.

Split mode runs ``llama-server`` on the hub and ``ggml-rpc-server`` on helpers.
Every machine must run the *same* build (the RPC protocol changes between
builds), so OllamaPower pins one release and refuses anything else.
"""
