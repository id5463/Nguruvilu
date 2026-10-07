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

# Stop a running instance before its file is replaced.
#
# A running turn keeps its conversation in memory until the turn ends, so the
# window is asked to close first — its close path stops the turn and persists
# what it produced. Only what refuses to leave is force-killed.
function Stop-Running {
    if (-not (Get-Process ngu-desktop -ErrorAction SilentlyContinue)) { return }

    if (-not ('NguruviluClose' -as [type])) {
        Add-Type -Namespace NguruviluPublish -Name WindowClose -MemberDefinition @'
[DllImport("user32.dll", CharSet = CharSet.Unicode)]
public static extern bool PostMessage(IntPtr hWnd, uint message, IntPtr wParam, IntPtr lParam);
'@
    }
    foreach ($proc in Get-Process ngu-desktop -ErrorAction SilentlyContinue) {
        if ($proc.MainWindowHandle -ne 0) {
            # WM_CLOSE: the same message the title bar's X sends.
            [NguruviluPublish.WindowClose]::PostMessage($proc.MainWindowHandle, 0x0010, [IntPtr]::Zero, [IntPtr]::Zero) | Out-Null
            Write-Host "  asked $($proc.Id) to close (its running turn persists first)"
        }
    }
    # The close path allows ~2s for a cancelled turn to land; wait a little
    # longer than that before insisting.
    for ($i = 0; $i -lt 30 -and (Get-Process ngu-desktop -ErrorAction SilentlyContinue); $i++) {
        Start-Sleep -Milliseconds 100
    }
    Get-Process ngu-desktop -ErrorAction SilentlyContinue | Stop-Process -Force
    Start-Sleep -Milliseconds 500
}

Step 'publishing to dist/'
# Both copies below are held open by a running instance, so it stops first.
Stop-Running
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
        # Stopped before the dist step above: a running instance holds both
        # copies open on Windows, and it is already gone by now.
        Publish (Join-Path $dist 'ngu-desktop.exe') $target
    }
}

Step 'done'
Write-Host "  artifacts in $dist"
