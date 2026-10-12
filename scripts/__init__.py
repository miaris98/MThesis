"""Operational entry points: training, evaluation, analysis, setup and sync.

Modules here are run as scripts (`python scripts/<area>/<name>.py`) but are also
importable as `scripts.<area>.<name>` so sibling scripts can reuse each other's
helpers. Each entry point puts the repository root on `sys.path` before importing
`src` or a sibling, so both invocation styles work.
"""
