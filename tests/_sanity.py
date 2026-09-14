"""A second tier of check, below a hard pytest assert.

A normal assert answers "is this correct?" - binary, and the only sound answer when the code
has an unambiguous contract (a shape, a raised exception, a sign). But some things this project
computes don't have a contract that crisp: "the target-speed head's prediction changed between
two images" is easy to assert on with `!=`, but a change of 1e-9 km/h technically satisfies that
assert while being indistinguishable from "the head isn't really looking." That's not a bug to
fail the build over - it's a judgment call a human should make, because the "right" threshold
depends on things a test can't know (how long the run trained, whether this is expected at
init time vs. after convergence, whether it's worth investigating today).

`soft_check` is for exactly that gap. It never fails the test - only `warnings.warn`s under
`LogicWarning`, which pytest collects into its normal warnings summary. Grep a full run's output
for "LogicWarning" to find what deserves a second look; each message should say what looked off
and why it isn't an automatic failure, so whoever reads it (a person, or a future Claude session)
can decide rather than re-deriving the judgment call from scratch.
"""
import warnings


class LogicWarning(UserWarning):
    """A soft sanity check came back suspicious. Not a test failure - a flag for human review."""


def soft_check(condition: bool, message: str) -> None:
    """Warn (never fail) when `condition` is False. `message` should explain what was expected,
    what was actually seen, and why this isn't being hard-asserted."""
    if not condition:
        warnings.warn(message, LogicWarning, stacklevel=2)
