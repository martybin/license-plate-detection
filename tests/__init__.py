"""Test package.

This file is not optional: several installed dependencies ship their own
top-level `tests` package, and Python prefers a regular package anywhere on
sys.path over a namespace package, so without an __init__.py here
`from tests.conftest import ...` resolves to site-packages instead of this
directory.
"""
