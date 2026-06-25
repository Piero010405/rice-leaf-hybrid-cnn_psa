#!/usr/bin/env bash
set -e

echo "Rice Leaf Hybrid CNN container started"
echo "Working directory: $(pwd)"
echo "Python version:"
python --version

exec "$@"
