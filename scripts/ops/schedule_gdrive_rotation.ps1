# Registers (or removes) a daily Windows scheduled task that runs scripts/ops/rotate_gdrive_token.sh.
# The task runs only while you are logged in (it needs your desktop to open the Google sign-in page, which asks for ONE click).
# usage:  powershell -File scripts\ops\schedule_gdrive_rotation.ps1 [-At 09:00]       register / update
#         powershell -File scripts\ops\schedule_gdrive_rotation.ps1 -Remove            delete the task
param([string]$At = "09:00", [switch]$Remove)
$name = "MThesis-GDrive-Token-Rotation"
if ($Remove) { Unregister-ScheduledTask -TaskName $name -Confirm:$false; "removed $name"; return }
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$bash = "C:\Program Files\Git\bin\bash.exe"
if (-not (Test-Path $bash)) { throw "Git Bash not found at $bash" }
$action = New-ScheduledTaskAction -Execute $bash -Argument "-lc `"cd '$($repo -replace '\\','/')' && bash scripts/ops/rotate_gdrive_token.sh --wait-min 20`"" -WorkingDirectory $repo
$trigger = New-ScheduledTaskTrigger -Daily -At $At
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Hours 1) -DontStopIfGoingOnBatteries -AllowStartIfOnBatteries
Register-ScheduledTask -TaskName $name -Action $action -Trigger $trigger -Settings $settings -Force | Out-Null
"registered $name daily at $At (runs only while logged in; log: $repo\.gdrive_rotation.log)"
