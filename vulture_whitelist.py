"""Confirmed Vulture false positives for interface parameters."""

# Context-manager protocol parameters are required by __exit__ even when the
# implementation intentionally ignores the exception details.
exc_type
exc_val
exc_tb
