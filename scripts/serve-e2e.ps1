# End-to-end gate check for `ngu-desktop --serve`.
#
# Every assertion here was a real hole at some point: an unauthenticated asset
# read through a Windows path, a rebinding page allowed because only the peer
# address was checked, a chunked body talking its way past the framing rules.
# The unit tests cover the pure functions; this covers the wiring, which is
# where those holes actually lived.
#
# Usage: pwsh -NoProfile -File scripts\serve-e2e.ps1 [-Exe <path>] [-Port 8899]
#
# Exit code 0 only if every check passed.

param(
    [string]$Exe = "",
    [int]$Port = 8899,
    [int]$StartupTimeoutSeconds = 20
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

if (-not $Exe) {
    $candidates = @(
        (Join-Path $PSScriptRoot "..\desktop\target\debug\ngu-desktop.exe"),
        (Join-Path $PSScriptRoot "..\desktop\target\release\ngu-desktop.exe")
    )
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate) { $Exe = (Resolve-Path -LiteralPath $candidate).Path; break }
    }
}
if (-not $Exe -or -not (Test-Path -LiteralPath $Exe)) {
    Write-Host "FAIL  no ngu-desktop.exe found; build it first (cargo build --manifest-path desktop\Cargo.toml)" -ForegroundColor Red
    exit 2
}

$token = "e2e-" + [guid]::NewGuid().ToString("N").Substring(0, 16)
$base = "http://127.0.0.1:$Port"

$failures = New-Object System.Collections.Generic.List[string]
$passes = 0

function Check([string]$name, [bool]$ok, [string]$detail = "") {
    if ($ok) {
        $script:passes++
        Write-Host ("PASS  " + $name) -ForegroundColor Green
    } else {
        $script:failures.Add($name + $(if ($detail) { " -- $detail" } else { "" }))
        Write-Host ("FAIL  " + $name + $(if ($detail) { " -- $detail" } else { "" })) -ForegroundColor Red
    }
}

# --- start the server ------------------------------------------------------

$log = Join-Path ([System.IO.Path]::GetTempPath()) ("ngu-serve-e2e-" + [guid]::NewGuid().ToString("N") + ".log")
$process = Start-Process -FilePath $Exe `
    -ArgumentList @("--serve", "127.0.0.1:$Port", "--token", $token, "--headless") `
    -RedirectStandardError $log -RedirectStandardOutput ($log + ".out") `
    -PassThru -WindowStyle Hidden

function Stop-Server {
    if ($process -and -not $process.HasExited) {
        try { $process.Kill($true) } catch { try { $process.Kill() } catch { } }
        try { $process.WaitForExit(5000) | Out-Null } catch { }
    }
}
trap { Stop-Server; throw }

# Wait for the port, not a fixed sleep: the binary prints its URL only after
# binding, and a slow machine must not turn into a flaky test.
$deadline = (Get-Date).AddSeconds($StartupTimeoutSeconds)
$listening = $false
while ((Get-Date) -lt $deadline) {
    if ($process.HasExited) { break }
    try {
        $probe = [System.Net.Sockets.TcpClient]::new()
        $probe.Connect("127.0.0.1", $Port)
        $probe.Close()
        $listening = $true
        break
    } catch { Start-Sleep -Milliseconds 150 }
}
if (-not $listening) {
    Stop-Server
    Write-Host "FAIL  server never listened on $Port (log: $log)" -ForegroundColor Red
    if (Test-Path -LiteralPath $log) { Get-Content -LiteralPath $log -Tail 20 | Write-Host }
    exit 2
}

$handler = [System.Net.Http.HttpClientHandler]::new()
$handler.AllowAutoRedirect = $false
$http = [System.Net.Http.HttpClient]::new($handler)
$http.Timeout = [TimeSpan]::FromSeconds(15)

function Send([string]$method, [string]$url, [hashtable]$headers = @{}, [string]$body = $null) {
    $request = [System.Net.Http.HttpRequestMessage]::new([System.Net.Http.HttpMethod]::new($method), $url)
    foreach ($key in $headers.Keys) {
        if ($key -eq "Host") { $request.Headers.Host = $headers[$key] }
        else { $request.Headers.TryAddWithoutValidation($key, $headers[$key]) | Out-Null }
    }
    if ($null -ne $body) {
        $request.Content = [System.Net.Http.StringContent]::new($body, [System.Text.Encoding]::UTF8, "application/json")
    }
    $response = $http.Send($request)
    $text = $response.Content.ReadAsStringAsync().GetAwaiter().GetResult()
    return [pscustomobject]@{
        Status = [int]$response.StatusCode
        Body   = $text
        Headers = $response.Headers
    }
}

# One request line, exactly as written, over a socket.
#
# `HttpClient` normalises `..` out of a URL before it sends anything, so the
# traversal checks below proved nothing through it: the server received `/x`.
# A check has to send what an attacker would send.
function SendRaw([string]$target) {
    $client = [System.Net.Sockets.TcpClient]::new()
    try {
        $client.Connect("127.0.0.1", $Port)
        $stream = $client.GetStream()
        $request = "GET $target HTTP/1.1`r`nHost: 127.0.0.1:$Port`r`nConnection: close`r`n`r`n"
        $bytes = [System.Text.Encoding]::ASCII.GetBytes($request)
        $stream.Write($bytes, 0, $bytes.Length)
        $stream.Flush()
        $reader = [System.IO.StreamReader]::new($stream)
        $text = $reader.ReadToEnd()
        $line = ($text -split "`r`n")[0]
        if ($line -match "\s(\d{3})\s") { return [int]$Matches[1] }
        return 0
    } finally {
        $client.Close()
    }
}

# Subscribe to the event stream, then make something happen, then read the
# stream until a frame with an `ev` field arrives.
#
# The server writes an event when one occurs and a keep-alive `: ping` in
# between, so "the stream is open" is not the same claim as "the stream
# delivers". Reading without triggering would only prove the ping.
function ReadEventAfterTrigger(
    [string]$streamUrl,
    [string]$commandUrl,
    [string]$body,
    [int]$milliseconds = 8000
) {
    $cancel = [System.Threading.CancellationTokenSource]::new()
    $cancel.CancelAfter($milliseconds)
    # Every field is always present: the caller runs under `Set-StrictMode` and
    # reading a property that only some paths set would throw instead of fail.
    $result = [pscustomobject]@{ Status = 0; Text = ""; Error = "" }
    # Its own client: the shared one may hand back a pooled connection the
    # server has already closed for a stream it was told to keep open.
    $handler = [System.Net.Http.HttpClientHandler]::new()
    $handler.AllowAutoRedirect = $false
    $client = [System.Net.Http.HttpClient]::new($handler)
    try {
        $request = [System.Net.Http.HttpRequestMessage]::new([System.Net.Http.HttpMethod]::Get, $streamUrl)
        $response = $client.Send($request, [System.Net.Http.HttpCompletionOption]::ResponseHeadersRead, $cancel.Token)
        $result.Status = [int]$response.StatusCode
        if ($result.Status -ne 200) { return $result }
        $stream = $response.Content.ReadAsStream($cancel.Token)

        # Subscribed first, so the event cannot be missed: the broadcast reaches
        # whoever is already listening.
        $command = [System.Net.Http.HttpRequestMessage]::new([System.Net.Http.HttpMethod]::Post, $commandUrl)
        $command.Content = [System.Net.Http.StringContent]::new($body, [System.Text.Encoding]::UTF8, "application/json")
        $null = $client.Send($command)

        $buffer = New-Object byte[] 4096
        $seen = ""
        while (-not $cancel.IsCancellationRequested) {
            $read = $stream.ReadAsync($buffer, 0, $buffer.Length, $cancel.Token).GetAwaiter().GetResult()
            if ($read -le 0) { break }
            $seen += [System.Text.Encoding]::UTF8.GetString($buffer, 0, $read)
            if ($seen.Contains('"ev"')) { break }
        }
        $result.Text = $seen
    } catch {
        # A stream that outlives the read is the normal case: what arrived before
        # the cancellation is the answer, so the text collected so far is kept.
        $result.Error = $_.Exception.Message
    } finally {
        $client.Dispose()
        $cancel.Dispose()
    }
    return $result
}

try {
    # --- no authentication -------------------------------------------------
    # The owner's call: the served shell answers everyone, because a page that
    # demanded a token answered 401 to the person who started it. What is left to
    # check is routing: the doors still answer by method and path alone.
    Check "GET / answers without a token" ((Send "GET" "$base/").Status -eq 200)
    Check "GET / answers any Host header" ((Send "GET" "$base/" @{ Host = "evil.example:$Port" }).Status -eq 200)
    Check "GET /cmd is not a page" ((Send "GET" "$base/cmd?token=$token").Status -eq 405)

    # --- URL-path traversal (the high-severity hole) -----------------------
    # `Path::join` with an absolute or backslash-climbing segment used to reach
    # any file on the machine, from a branch that was outside the auth check.
    Check "GET /C:\Windows\win.ini is refused" ((SendRaw "/C:\Windows\win.ini?token=$token") -eq 404)
    Check "GET /..\..\x is refused" ((SendRaw "/..\..\x?token=$token") -eq 404)
    # Percent-encoded, because a server that decodes before it checks is the
    # shape this hole usually arrives in.
    Check "GET /%2e%2e%2fx is refused" ((SendRaw "/%2e%2e%2fx?token=$token") -eq 404)
    Check "GET /a\..\..\b is refused" ((SendRaw "/a\..\..\b?token=$token") -eq 404)
    Check "GET /..%5C..%5Cx is refused" ((SendRaw "/..%5C..%5Cx?token=$token") -eq 404)

    # --- query handling ----------------------------------------------------
    # A repeated `token=` used to mean "two credentials in one request" and was
    # refused; with no credential in play it is ordinary query data, and the page
    # must still load.
    Check "a repeated token query is ordinary data" ((Send "GET" "$base/?token=$token&token=$token").Status -eq 200)

    # --- method and framing ------------------------------------------------
    $chunked = [System.Net.Http.HttpRequestMessage]::new([System.Net.Http.HttpMethod]::Post, "$base/cmd?token=$token")
    $chunked.Content = [System.Net.Http.ByteArrayContent]::new([System.Text.Encoding]::UTF8.GetBytes('{"cmd":"status"}'))
    $chunked.Headers.TransferEncodingChunked = $true
    $chunkedResponse = $http.Send($chunked)
    Check "Transfer-Encoding: chunked is refused" ([int]$chunkedResponse.StatusCode -eq 501) ([string][int]$chunkedResponse.StatusCode)

    # An ordinary command still works, and answers with JSON.
    $status = Send "POST" "$base/cmd?token=$token" @{ "Content-Type" = "application/json" } '{"cmd":"status"}'
    Check "POST /cmd with the token answers 200" ($status.Status -eq 200) ("status " + $status.Status)

    # --- the event stream --------------------------------------------------
    # `id:` is what lets a reconnecting page tell a gap from a repeat, and a
    # subscriber has to receive the events a command produces — not just the
    # keep-alive that proves the socket is open.
    $events = ReadEventAfterTrigger "$base/events?token=$token" "$base/cmd?token=$token" '{"cmd":"status"}'
    Check "GET /events streams" ($events.Status -eq 200) ("status " + $events.Status + " " + $events.Error)
    Check "GET /events delivers the event a command produced" ($events.Text.Contains('"ev"')) ("first bytes: " + $events.Text)
    $noToken = ReadEventAfterTrigger "$base/events" "$base/cmd" '{"cmd":"status"}'
    Check "GET /events streams without a token" ($noToken.Status -eq 200) ("status " + $noToken.Status + " " + $noToken.Error)

    # --- trailing detail ---------------------------------------------------
    # An unknown page path is answered with the page, because the shell is a
    # single page that routes on its own. That is true for every caller now: the
    # path decides, not who is asking.
    $loopback = Send "GET" "$base/nope" @{ Host = "127.0.0.1:$Port" }
    Check "an unknown path serves the page to this machine" ($loopback.Status -eq 200) ("status " + $loopback.Status)
    $foreign = Send "GET" "$base/nope" @{ Host = "evil.example:$Port" }
    Check "an unknown path serves the page to any Host too" ($foreign.Status -eq 200) ("status " + $foreign.Status)
} finally {
    $http.Dispose()
    Stop-Server
}

Write-Host ""
if ($failures.Count -eq 0) {
    Write-Host "serve gate: $passes checks passed" -ForegroundColor Green
    exit 0
}
Write-Host "serve gate: $passes passed, $($failures.Count) failed" -ForegroundColor Red
foreach ($failure in $failures) { Write-Host "  - $failure" -ForegroundColor Red }
Write-Host "server log: $log"
exit 1
