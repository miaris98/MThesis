param([string]$Transit, [string]$Day)
foreach ($d in Get-ChildItem $Transit -Directory) {
  if (-not (Get-ChildItem $d.FullName -Recurse -File)) { continue }
  $dest = "E:\MThesis_EXP\live_${Day}_$($d.Name)"
  if (-not (Test-Path "E:\MThesis_EXP")) { Write-Output "E: UNAVAILABLE (E:\MThesis_EXP not found)"; exit 1 }
  New-Item -ItemType Directory -Force -Path $dest | Out-Null
  robocopy $d.FullName $dest /E /NFL /NDL /NJH /NJS /NP | Out-Null
  $bad = 0; $n = 0
  foreach ($f in Get-ChildItem $d.FullName -Recurse -File) {
    $n++
    $t = Join-Path $dest $f.FullName.Substring($d.FullName.Length).TrimStart('\')
    if (-not (Test-Path -LiteralPath $t) -or (Get-Item -LiteralPath $t).Length -ne $f.Length) { $bad++ }
  }
  Write-Output "E: $($d.Name): $n pulled files, missing/size-mismatch on E: $bad $(if ($bad -eq 0) {'MATCH'} else {'WRITE FAILED/MISMATCH'})"
}
