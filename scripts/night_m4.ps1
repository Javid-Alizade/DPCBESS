# Milestone M4 night chain (docs/decisions.md, "CHANGE OF M4 PLAN ITEM 7"):
#   1. the lossless 2-day pilot, run once, with the pre-specified pass criteria (exit code 1 = a criterion failed: stop);
#   2. if it passes: the full persistence MPC on all validation steps, both regimes, resumable;
#   3. then the oracle subset (every 4th full validation day), all controllers under identical conditions.
# Run from the repository root:  powershell -File scripts\night_m4.ps1
# Running it again after an interruption continues where it stopped: the pilot file already exists and passed, the full run
# resumes from its checkpoint, and finished oracle parts are skipped.
# While it runs, the computer is kept from idle sleep (SetThreadExecutionState, released when this process ends; no setting
# is changed).
$ErrorActionPreference = "Stop"
Add-Type -Namespace Win32 -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint flags);'
[Win32.Power]::SetThreadExecutionState([uint32]"0x80000001") | Out-Null  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
$python = ".\.venv\Scripts\python.exe"
$pilot = "results\m4_mpc_pilot_lossless.json"

if (Test-Path $pilot) {
    $passed = (Get-Content $pilot -Raw | ConvertFrom-Json).all_criteria_pass
    Write-Output "pilot already run once; all criteria pass: $passed"
} else {
    & $python -I scripts\pilot_mpc_lossless_m4.py
    $passed = ($LASTEXITCODE -eq 0)
}
if (-not $passed) { Write-Output "STOP: a pilot criterion failed; nothing else is run."; exit 1 }

& $python -I scripts\run_mpc_full_m4.py
if ($LASTEXITCODE -ne 0) { Write-Output "STOP: the full run failed."; exit 1 }

& $python -I scripts\run_oracle_subset_m4.py
if ($LASTEXITCODE -ne 0) { Write-Output "STOP: the oracle subset failed."; exit 1 }
Write-Output "night chain finished"
