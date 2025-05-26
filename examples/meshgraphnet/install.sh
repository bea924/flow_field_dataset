#! /bin/bash

UV_HTTP_TIMEOUT=180 uv pip install  dgl -f "https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html"
uv pip install -r requirements.txt

# # Install the dgl backend for pytorch
# uv pip install dgl-cu121 -f "https://data.dgl.ai/wheels/torch-2.4/cu121/repo.html"