#Requires -Version 5.1
# Isolated checks: only fixture data below this record directory can be removed.
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'Cleanup-Codex.ps1')

function Assert-Check {
    param([bool]$Condition, [string]$Message)
    if (-not $Condition) { throw $Message }
}

function Assert-Refused {
    param([scriptblock]$Action, [string]$Message)
    $refused = $false
    try { & $Action | Out-Null } catch { $refused = $true }
    Assert-Check $refused $Message
}

$fixtureParent = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '.verification'))
$fixtureRoot = [IO.Path]::GetFullPath((Join-Path $fixtureParent ([guid]::NewGuid().ToString('N'))))
Assert-Check ($fixtureRoot.StartsWith($fixtureParent + '\', [StringComparison]::OrdinalIgnoreCase)) 'Unsafe fixture root.'
$testResults = New-Object 'System.Collections.Generic.List[string]'

try {
    $profileRoot = Join-Path $fixtureRoot 'Profile [test]'
    $localRoot = Join-Path $profileRoot 'AppData\Local'
    $roamingRoot = Join-Path $profileRoot 'AppData\Roaming'
    $fixtureTargets = @(Get-CodexCleanupTargets -ProfileRoot $profileRoot -LocalAppDataRoot $localRoot -RoamingAppDataRoot $roamingRoot)
    Assert-Check ($fixtureTargets.Count -eq 5) 'Expected exactly five cleanup targets.'
    foreach ($target in $fixtureTargets) {
        $nestedPath = Join-Path $target.Path 'nested'
        New-Item -ItemType Directory -Path $nestedPath -Force | Out-Null
        Set-Content -LiteralPath (Join-Path $nestedPath 'evidence.txt') -Value 'fixture only'
    }
    $neighborFiles = @(
        (Join-Path $localRoot 'OpenAI\OtherApp\keep.txt'),
        (Join-Path $localRoot 'Packages\Other.Package\keep.txt'),
        (Join-Path $profileRoot 'project\keep.txt')
    )
    foreach ($neighborFile in $neighborFiles) {
        New-Item -ItemType Directory -Path ([IO.Path]::GetDirectoryName($neighborFile)) -Force | Out-Null
        Set-Content -LiteralPath $neighborFile -Value 'must survive'
    }

    # Preview and cancel use fixtures even though they exercise the real entry function.
    $script:VerificationTargets = $fixtureTargets
    function Get-CodexCleanupTargets { param($ProfileRoot, $LocalAppDataRoot, $RoamingAppDataRoot) $script:VerificationTargets }
    function Assert-CodexUninstalled { }
    function Read-Host { param($Prompt) 'cancel' }
    Assert-Check ((Invoke-CodexCleanup -Preview) -eq 0) 'Preview failed.'
    Assert-Check ((Invoke-CodexCleanup) -eq 0) 'Cancel failed.'
    foreach ($target in $fixtureTargets) {
        Assert-Check (Test-Path -LiteralPath (Join-Path $target.Path 'nested\evidence.txt')) 'Preview or cancel modified data.'
    }
    $testResults.Add('PASS: Preview and cancel leave all fixture data unchanged.')

    foreach ($target in $fixtureTargets) {
        Assert-Check ((Remove-CodexCleanupTarget -Target $target) -eq 'Removed') 'Expected fixture removal.'
        Assert-Check (-not (Test-Path -LiteralPath $target.Path)) 'Fixture target still exists.'
        Assert-Check ((Remove-CodexCleanupTarget -Target $target) -eq 'Missing') 'Repeated cleanup must skip missing targets.'
    }
    foreach ($neighborFile in $neighborFiles) {
        Assert-Check (Test-Path -LiteralPath $neighborFile) 'Unrelated data was removed.'
    }
    $testResults.Add('PASS: All five targets removed, repeat cleanup skips them, and unrelated neighboring data survives.')

    $badTarget = [pscustomobject]@{ Key = 'Runtime'; Root = $localRoot; Path = (Join-Path $localRoot 'OpenAI') }
    Assert-Refused { Remove-CodexCleanupTarget -Target $badTarget } 'Parent directory must be refused.'
    $badTarget = [pscustomobject]@{ Key = 'Home'; Root = $profileRoot; Path = (Join-Path $profileRoot '..\outside') }
    Assert-Refused { Remove-CodexCleanupTarget -Target $badTarget } 'Escaped target must be refused.'
    $badTarget = [pscustomobject]@{ Key = 'Unknown'; Root = $profileRoot; Path = (Join-Path $profileRoot '.codex') }
    Assert-Refused { Remove-CodexCleanupTarget -Target $badTarget } 'Unknown target must be refused.'
    $badTarget = [pscustomobject]@{ Key = 'Home'; Root = 'C:\'; Path = 'C:\.codex' }
    Assert-Refused { Remove-CodexCleanupTarget -Target $badTarget } 'Volume root must be refused.'
    $badTarget = [pscustomobject]@{ Key = 'Home'; Root = 'C:relative'; Path = 'C:relative\.codex' }
    Assert-Refused { Remove-CodexCleanupTarget -Target $badTarget } 'Drive-relative paths must be refused.'
    $testResults.Add('PASS: Parent, escaped, unknown, volume-root, and drive-relative paths are refused.')

    $outsideDirectory = Join-Path $fixtureRoot 'outside'
    New-Item -ItemType Directory -Path $outsideDirectory -Force | Out-Null
    $outsideSentinel = Join-Path $outsideDirectory 'keep.txt'
    Set-Content -LiteralPath $outsideSentinel -Value 'must survive junction checks'
    $junctionPaths = New-Object 'System.Collections.Generic.List[string]'
    try {
        # Root reparse point: reject without traversing into its target.
        New-Item -ItemType Junction -Path $fixtureTargets[0].Path -Target $outsideDirectory | Out-Null
        $junctionPaths.Add($fixtureTargets[0].Path)
        Assert-Refused { Remove-CodexCleanupTarget -Target $fixtureTargets[0] } 'Target junction must be refused.'

        # Nested reparse point: scanning must reject the entire cleanup target.
        New-Item -ItemType Directory -Path $fixtureTargets[1].Path -Force | Out-Null
        $nestedJunction = Join-Path $fixtureTargets[1].Path 'linked-data'
        New-Item -ItemType Junction -Path $nestedJunction -Target $outsideDirectory | Out-Null
        $junctionPaths.Add($nestedJunction)
        Assert-Refused { Remove-CodexCleanupTarget -Target $fixtureTargets[1] } 'Nested junction must be refused.'

        # Ancestor reparse point: a redirected root is also refused.
        $linkedRoot = Join-Path $fixtureRoot 'linked-profile'
        New-Item -ItemType Junction -Path $linkedRoot -Target $outsideDirectory | Out-Null
        $junctionPaths.Add($linkedRoot)
        $linkedTarget = [pscustomobject]@{ Key = 'Home'; Root = $linkedRoot; Path = (Join-Path $linkedRoot '.codex') }
        Assert-Refused { Remove-CodexCleanupTarget -Target $linkedTarget } 'Ancestor junction must be refused.'
        Assert-Check (Test-Path -LiteralPath $outsideSentinel) 'Junction target data was modified.'
        $testResults.Add('PASS: Target, nested, and ancestor junctions are refused; external sentinel survives.')
    }
    finally {
        # Delete junction entries nonrecursively before cleaning fixture directories.
        foreach ($junctionPath in $junctionPaths) {
            Assert-Check ($junctionPath.StartsWith($fixtureRoot + '\', [StringComparison]::OrdinalIgnoreCase)) 'Unsafe junction cleanup.'
            if (Test-Path -LiteralPath $junctionPath) { [IO.Directory]::Delete($junctionPath, $false) }
        }
    }

    # Reload original preflight function, then mock only the Windows query results.
    . (Join-Path $PSScriptRoot 'Cleanup-Codex.ps1')
    function Get-AppxPackage { param($Name) [pscustomobject]@{ Name = 'OpenAI.Codex' } }
    Assert-Refused { Assert-CodexUninstalled } 'Installed app must prevent deletion.'
    function Get-AppxPackage { param($Name) throw 'Simulated package query failure.' }
    Assert-Refused { Assert-CodexUninstalled } 'Unknown uninstall state must prevent deletion.'
    function Get-AppxPackage { param($Name) }
    function Get-Process { [pscustomobject]@{ ProcessName = 'codex' } }
    Assert-Refused { Assert-CodexUninstalled } 'Running process must prevent deletion.'
    function Get-Process { }
    Assert-CodexUninstalled
    $testResults.Add('PASS: Installed app, package-query failure, and active process block cleanup; clear state passes.')

    $testResults | Write-Output
    Write-Output ('All isolated checks passed on PowerShell ' + $PSVersionTable.PSVersion.ToString())
}
finally {
    $resolvedFixtureRoot = [IO.Path]::GetFullPath($fixtureRoot)
    if (-not $resolvedFixtureRoot.StartsWith($fixtureParent + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Refusing unsafe fixture cleanup.'
    }
    if (Test-Path -LiteralPath $resolvedFixtureRoot) {
        Remove-Item -LiteralPath $resolvedFixtureRoot -Recurse -Force -ErrorAction Stop
    }
    if ((Test-Path -LiteralPath $fixtureParent) -and
        @(Get-ChildItem -LiteralPath $fixtureParent -Force).Count -eq 0) {
        Remove-Item -LiteralPath $fixtureParent -Force -ErrorAction Stop
    }
}
