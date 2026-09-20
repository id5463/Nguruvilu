# Build every artifact and publish it where it is actually run from.
#
# This exists because copying binaries by hand went wrong: a build succeeded, the
# copy silently failed, and the stale executable on the desktop was missing a
# feature that had already shipped. A publish step that cannot tell whether it
# worked is not a publish step.
#
# Every copy is verified by hash, and the script fails loudly if any of them
# disagree. Run it from the repository root:
#
#     pwsh -File release.ps1
#
# Pass -SkipDesktop to leave the desktop shortcut copy alone.

[CmdletBinding()]
param(
    # Also place ngu-desktop.exe on the user's desktop.
    [switch]$SkipDesktop
)

$ErrorActionPreference = 'Stop'

$root = $PSScriptRoot
$cargo = if ($env:CARGO_BIN) { $env:CARGO_BIN } else { 'cargo' }
$dist = Join-Path $root 'dist'

function Step($message) {
    Write-Host ""
    Write-Host "== $message" -ForegroundColor Cyan
}

function Fail($message) {
    Write-Host "FAILED: $message" -ForegroundColor Red
    exit 1
}

# A copied file is only published once its hash matches the source.
function Publish($source, $target) {
    if (-not (Test-Path $source)) { Fail "missing build output: $source" }
    Copy-Item $source $target -Force
    $a = (Get-FileHash $source -Algorithm SHA256).Hash
    $b = (Get-FileHash $target -Algorithm SHA256).Hash
    if ($a -ne $b) { Fail "copy did not land: $target" }
    $size = [math]::Round((Get-Item $target).Length / 1KB, 1)
    Write-Host ("  {0,-26} {1,8} KB  {2}" -f (Split-Path $target -Leaf), $size, $a.Substring(0, 12))
}

Step 'building the kernel and CLI'
& $cargo build --release --manifest-path (Join-Path $root 'Cargo.toml')
if ($LASTEXITCODE -ne 0) { Fail 'kernel build' }

Step 'building the desktop shell'
& $cargo build --release --manifest-path (Join-Path $root 'desktop/Cargo.toml')
if ($LASTEXITCODE -ne 0) { Fail 'desktop build' }

Step 'building the example plugin'
$pluginManifest = Join-Path $root 'examples/demo-plugin/Cargo.toml'
if (Test-Path $pluginManifest) {
    & $cargo build --release --manifest-path $pluginManifest
    if ($LASTEXITCODE -ne 0) { Fail 'plugin build' }
}

Step 'publishing to dist/'
New-Item -ItemType Directory -Path $dist -Force | Out-Null
Publish (Join-Path $root 'target/release/ngu.exe') (Join-Path $dist 'ngu.exe')
Publish (Join-Path $root 'desktop/target/release/ngu-desktop.exe') (Join-Path $dist 'ngu-desktop.exe')
$plugin = Join-Path $root 'examples/demo-plugin/target/release/ngu_demo_plugin.dll'
if (Test-Path $plugin) {
    Publish $plugin (Join-Path $dist 'ngu_demo_plugin.dll')
}

if (-not $SkipDesktop) {
    Step 'publishing to the desktop'
    $desktop = [Environment]::GetFolderPath('Desktop')
    if ([string]::IsNullOrWhiteSpace($desktop)) {
        Write-Host "  no desktop folder; skipping" -ForegroundColor Yellow
    } else {
        $target = Join-Path $desktop 'ngu-desktop.exe'
        # A running instance holds the file open on Windows.
        Get-Process ngu-desktop -ErrorAction SilentlyContinue | Stop-Process -Force
        Start-Sleep -Milliseconds 500
        Publish (Join-Path $dist 'ngu-desktop.exe') $target
    }
}

Step 'done'
Write-Host "  artifacts in $dist"
