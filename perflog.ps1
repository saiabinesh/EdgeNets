# Create log folder if it doesn't exist
$logPath = 'C:\logs'
if (!(Test-Path $logPath)) { New-Item -ItemType Directory -Path $logPath | Out-Null }

# Write CSV header if file is new
$csvFile = Join-Path $logPath 'perf_log.csv'
if (!(Test-Path $csvFile)) {
    "Timestamp,GPU Util (%),GPU Mem Used (MiB),Disk Bytes/sec,Disk Reads/sec,Disk Writes/sec" |
      Out-File -FilePath $csvFile -Encoding utf8
}

# Infinite loop: sample every 900 seconds (15 mins)
while ($true) {
    $ts = Get-Date -Format o

    # GPU stats via nvidia-smi
    $gpuLine = & nvidia-smi --query-gpu=utilization.gpu,memory.used `
              --format=csv,noheader,nounits
    # this returns something like "23 %, 512 MiB" so split on comma:
    $gpuStats = $gpuLine -split ',' | ForEach-Object { $_.Trim() }

    # Disk stats via perf counters
    $counterPaths = @(
      '\PhysicalDisk(_Total)\Disk Bytes/sec',
      '\PhysicalDisk(_Total)\Disk Reads/sec',
      '\PhysicalDisk(_Total)\Disk Writes/sec'
    )
    $counters = Get-Counter -Counter $counterPaths
    $diskValues = $counters.CounterSamples |
                  Sort-Object Path |
                  ForEach-Object { [math]::Round($_.CookedValue,2) }

    # Build CSV line
    $csvValues = @(
      $ts,
      $gpuStats[0],
      $gpuStats[1]
    ) + $diskValues

    ($csvValues -join ',') | Out-File -FilePath $csvFile -Append -Encoding utf8

    Start-Sleep -Seconds 900
}
