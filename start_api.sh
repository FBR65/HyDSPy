#!/bin/bash

# Starte die API mit uv run
uv run uvicorn src.api.app:app --host 0.0.0.0 --port 8000