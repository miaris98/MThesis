"""TODO (2026-11-03): every trainer flag that a launch script passes must be a real option of the trainer's parser. A stale or misspelled flag
in a box script otherwise surfaces only when the box starts (the 2026-11-03 Gate 1 launch failed that way). The check reads each shell script
that runs train_ez_offpolicy.py: it joins backslash-continued lines, splits each line into commands (; && || |), keeps the commands that run
the trainer, and adds the flags of any variable those commands expand (ARM="--trunk ..."). Eval commands in the same script are excluded."""
import glob
import os
import re

from atari_qwen.training.train_ez_offpolicy import build_parser

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRAINER = "train_ez_offpolicy.py"
FLAG = re.compile(r"(?<![\w-])(--[a-z][a-z0-9_-]*)")
ASSIGN = re.compile(r"^\s*([A-Za-z_]\w*)=[\"']?(.*)$")
REF = re.compile(r"\$\{?([A-Za-z_]\w*)")


def trainer_flags_in(text: str) -> set:
    lines = text.replace("\\n", " ").splitlines()
    commands = [c for line in lines for c in re.split(r";|&&|\|\||\|", line)]
    trainer_cmds = [c for c in commands if TRAINER in c]
    refs = {v for c in trainer_cmds for v in REF.findall(c)}
    values = []
    for line in lines:
        m = ASSIGN.match(line)
        if m and m.group(1) in refs:
            values.append(m.group(2))
    flags = set()
    for c in trainer_cmds + values:
        flags.update(FLAG.findall(c))
    return flags


def trainer_flags_by_script():
    out = {}
    for path in glob.glob(os.path.join(ROOT, "scripts", "**", "*.sh"), recursive=True):
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()
        for flag in trainer_flags_in(text):
            out.setdefault(flag, set()).add(os.path.relpath(path, ROOT))
    return out


def test_the_scripts_pass_only_flags_the_trainer_knows():
    known = set(build_parser()._option_string_actions)
    used = trainer_flags_by_script()
    assert used, "no launch script found: the contract test would pass vacuously"
    unknown = {flag: sorted(files) for flag, files in used.items() if flag not in known}
    assert not unknown, f"flags the trainer does not accept: {unknown}"


def test_the_gate_command_is_covered_by_the_contract():
    used = trainer_flags_by_script()
    assert "--reanalyze-merge" in used and "--mixer-where" in used and "--mixer-qk-norm" in used


def test_eval_flags_in_the_same_script_are_not_trainer_flags():
    text = 'eval_loop_stats.py $ck --episodes 30 --sims 16 \\n  --max-steps 27000\npython -u train_ez_offpolicy.py --trunk conv_transformer\n'
    assert trainer_flags_in(text) == {"--trunk"}


def test_variables_expanded_by_a_trainer_command_are_checked():
    text = 'ARM="--trunk qwen --mixer-size small"\npython -u train_ez_offpolicy.py $ARM --seed 1\n'
    assert trainer_flags_in(text) == {"--trunk", "--mixer-size", "--seed"}
