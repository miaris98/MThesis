"""Closed-loop (in-simulator) evaluation for the World on Rails policies.

Separate from `src/training/wor_eval.py`, which is the open-loop held-out pass over a
fixed dataset. Everything in Challenge Group 13 up to 13.29 is an open-loop number;
this package exists to produce the closed-loop ones.
"""
