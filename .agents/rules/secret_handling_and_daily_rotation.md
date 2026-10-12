# Secret Handling, Daily API Key Rotation & Vast.ai Access Rules

## 1. Strictly Never Read `.env` or Dotfiles (`.*`)
- **NEVER** use `view_file`, `grep_search`, or shell commands (`cat`, `type`, `Get-Content`, `head`, `tail`, etc.) to directly read any `.env` files or any files whose names start with a dot `.` (e.g. `.env`, `.env.local`, `.vast_api_key`, `.remote`, etc.).
- **Always assume credentials in dotfiles exist and work correctly**.
- If any operational information, clarification, or credential verification is needed regarding them, **always ask the user directly** instead of reading the file.

## 2. Daily API Key Reroll Prompting
- At the start of every new calendar day / new daily session, **prompt the user to reroll/rotate their API keys** (Vast.ai, Hugging Face, etc.) to maintain zero-leak security hygiene.

## 3. Vast.ai Platform Integration & Autonomous Fleet Queries
- Both Antigravity and Claude can directly query and orchestrate the Vast.ai fleet via the local CLI:
  - Command: `python scripts/setup/vast_fleet.py list` or `vastai show instances`
  - Switching active box in `.remote`: `python scripts/setup/vast_fleet.py select <idx_or_id>`
  - Searching GPU marketplace: `vastai search offers 'gpu_name=RTX_3090 ...'`
- Authentication is stored in user profile config (`~/.config/vastai/vast_api_key` & `vast_tfa_key`), so no API tokens ever need to be read or passed in command arguments.
