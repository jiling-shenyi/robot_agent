#Requires -Version 5.1
<#
Codex 卸载后残留清理。双击同目录 Cleanup-Codex.cmd 运行。
只读预览：powershell.exe -NoProfile -File .\Cleanup-Codex.ps1 -PreviewOnly
不卸载程序、不结束进程、不自动提权，也不清理项目代码。
#>
[CmdletBinding()]
param([switch]$PreviewOnly)

$ErrorActionPreference = 'Stop'

function Get-CodexCleanupTargets {
    param(
        [Parameter(Mandatory = $true)][string]$ProfileRoot,
        [Parameter(Mandatory = $true)][string]$LocalAppDataRoot,
        [Parameter(Mandatory = $true)][string]$RoamingAppDataRoot
    )

    $definitions = @(
        @{ Key = 'Home'; Root = $ProfileRoot; Relative = '.codex'; Label = '用户配置、历史、技能和生成文件' },
        @{ Key = 'Runtime'; Root = $LocalAppDataRoot; Relative = 'OpenAI\Codex'; Label = '命令行组件及运行环境' },
        @{ Key = 'Local'; Root = $LocalAppDataRoot; Relative = 'Codex'; Label = '本地日志' },
        @{ Key = 'Roaming'; Root = $RoamingAppDataRoot; Relative = 'Codex'; Label = '桌面应用数据' },
        @{ Key = 'Package'; Root = $LocalAppDataRoot; Relative = 'Packages\OpenAI.Codex_2p2nqsd0c76g0'; Label = '桌面程序包用户数据残留' }
    )

    foreach ($definition in $definitions) {
        if ([string]::IsNullOrWhiteSpace($definition.Root) -or
            $definition.Root -notmatch '^[A-Za-z]:[\\/]') {
            throw '无法确定当前用户的绝对数据目录，停止清理。'
        }
        $rootPath = [IO.Path]::GetFullPath($definition.Root).TrimEnd('\', '/')
        [pscustomobject]@{
            Key = $definition.Key
            Root = $rootPath
            Path = [IO.Path]::GetFullPath([IO.Path]::Combine($rootPath, $definition.Relative))
            Label = $definition.Label
        }
    }
}

function Assert-CodexCleanupTarget {
    param([Parameter(Mandatory = $true)]$Target, [switch]$ScanContents)

    # The allowlist is reconstructed here rather than trusting a supplied path.
    $allowedRelativePaths = @{
        Home = '.codex'
        Runtime = 'OpenAI\Codex'
        Local = 'Codex'
        Roaming = 'Codex'
        Package = 'Packages\OpenAI.Codex_2p2nqsd0c76g0'
    }
    if (-not $allowedRelativePaths.ContainsKey([string]$Target.Key)) {
        throw '目标不在固定清理列表中。'
    }
    if ([string]::IsNullOrWhiteSpace($Target.Root) -or
        [string]::IsNullOrWhiteSpace($Target.Path) -or
        $Target.Root -notmatch '^[A-Za-z]:[\\/]' -or
        $Target.Path -notmatch '^[A-Za-z]:[\\/]') {
        throw '目标路径必须是明确的绝对路径。'
    }
    $rootPath = [IO.Path]::GetFullPath($Target.Root).TrimEnd('\', '/')
    $volumeRoot = [IO.Path]::GetPathRoot($rootPath).TrimEnd('\', '/')
    if ($rootPath -eq $volumeRoot -or $rootPath.StartsWith('\\')) {
        throw '拒绝使用磁盘根目录或网络共享作为用户数据根目录。'
    }
    $actualPath = [IO.Path]::GetFullPath($Target.Path).TrimEnd('\', '/')
    $expectedPath = [IO.Path]::GetFullPath(
        [IO.Path]::Combine($rootPath, $allowedRelativePaths[$Target.Key])
    ).TrimEnd('\', '/')
    if (-not $actualPath.Equals($expectedPath, [StringComparison]::OrdinalIgnoreCase) -or
        -not $actualPath.StartsWith($rootPath + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw ('目标超出固定清理范围：' + $Target.Path)
    }

    # Reject links in the complete ancestor chain, including a relocated profile.
    $ancestorPath = $actualPath
    while (-not [string]::IsNullOrEmpty($ancestorPath)) {
        if (Test-Path -LiteralPath $ancestorPath -ErrorAction Stop) {
            $ancestor = Get-Item -LiteralPath $ancestorPath -Force -ErrorAction Stop
            if (($ancestor.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
                throw ('检测到符号链接或目录联接，拒绝清理：' + $ancestorPath)
            }
            if (-not $ancestor.PSIsContainer) {
                throw ('目标或父路径不是目录：' + $ancestorPath)
            }
        }
        $parentInfo = [IO.Directory]::GetParent($ancestorPath)
        if ($null -eq $parentInfo) { break }
        $ancestorPath = $parentInfo.FullName
    }

    if ($ScanContents -and (Test-Path -LiteralPath $actualPath -ErrorAction Stop)) {
        $pendingDirectories = New-Object 'System.Collections.Generic.Stack[string]'
        $pendingDirectories.Push($actualPath)
        while ($pendingDirectories.Count -gt 0) {
            $directoryPath = $pendingDirectories.Pop()
            foreach ($entry in Get-ChildItem -LiteralPath $directoryPath -Force -ErrorAction Stop) {
                if (($entry.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
                    throw ('目录内含符号链接或目录联接，拒绝清理此目标：' + $entry.FullName)
                }
                if ($entry.PSIsContainer) {
                    $pendingDirectories.Push($entry.FullName)
                }
            }
        }
    }
}

function Remove-CodexCleanupTarget {
    param([Parameter(Mandatory = $true)]$Target)

    Assert-CodexCleanupTarget -Target $Target -ScanContents
    $verifiedPath = [IO.Path]::GetFullPath($Target.Path)
    if (-not (Test-Path -LiteralPath $verifiedPath -ErrorAction Stop)) {
        return 'Missing'
    }
    # The resolved absolute target and its subtree have been checked above.
    Remove-Item -LiteralPath $verifiedPath -Recurse -Force -ErrorAction Stop
    if (Test-Path -LiteralPath $verifiedPath -ErrorAction Stop) {
        throw ('删除后目录仍存在：' + $verifiedPath)
    }
    return 'Removed'
}

function Assert-CodexUninstalled {
    try {
        $installedPackages = @(Get-AppxPackage -Name 'OpenAI.Codex' -ErrorAction Stop)
    }
    catch {
        throw ('无法确认 Codex 已卸载。请在独立的 Windows PowerShell 中运行此脚本。详情：' + $_.Exception.Message)
    }
    if ($installedPackages.Count -gt 0) {
        throw '检测到当前用户仍安装 Codex。请先在 Windows 设置的“已安装的应用”中卸载，再运行清理。'
    }
    $activeProcesses = @(Get-Process -ErrorAction Stop | Where-Object {
        $_.ProcessName -match '^(Codex|codex-app-server|codex-code-mode-host|codex-command-runner)$'
    })
    if ($activeProcesses.Count -gt 0) {
        $processNames = ($activeProcesses | Select-Object -ExpandProperty ProcessName -Unique) -join ', '
        throw ('相关进程仍在运行，请关闭后重试；必要时先重启电脑：' + $processNames)
    }
}

function Invoke-CodexCleanup {
    param([switch]$Preview)

    Write-Host ''
    Write-Host 'Codex 卸载后残留清理' -ForegroundColor Cyan
    Write-Host '只处理以下五个当前用户目录，不处理项目代码或其他 OpenAI 应用。'
    $targets = @(Get-CodexCleanupTargets `
        -ProfileRoot ([Environment]::GetFolderPath('UserProfile')) `
        -LocalAppDataRoot ([Environment]::GetFolderPath('LocalApplicationData')) `
        -RoamingAppDataRoot ([Environment]::GetFolderPath('ApplicationData')))

    foreach ($target in $targets) {
        Assert-CodexCleanupTarget -Target $target
        $status = if (Test-Path -LiteralPath $target.Path -ErrorAction Stop) { '存在' } else { '不存在，跳过' }
        Write-Host ('[{0}] {1}' -f $status, $target.Path)
        Write-Host ('  ' + $target.Label)
    }
    if ($Preview) {
        Write-Host '预览结束：没有删除任何文件。' -ForegroundColor Green
        return 0
    }

    Assert-CodexUninstalled
    Write-Host ''
    Write-Host '警告：将永久删除上述目录，不进入回收站。' -ForegroundColor Yellow
    Write-Host '.codex 内的本地聊天、配置、技能、插件、生成文件及工作副本也会删除。'
    Write-Host '这些数据可能与命令行版和编辑器扩展共用。请先确认需要的内容已备份。'
    $answer = Read-Host '确认已卸载、已备份并要清理时，输入大写 DELETE；其他输入取消'
    if ($answer -cne 'DELETE') {
        Write-Host '已取消，没有删除任何文件。'
        return 0
    }

    # Recheck after confirmation, in case the app was opened while waiting.
    Assert-CodexUninstalled
    $reportLines = New-Object 'System.Collections.Generic.List[string]'
    $reportLines.Add(('Codex 残留清理结果：' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss')))
    $removedCount = 0
    $missingCount = 0
    $failureCount = 0
    foreach ($target in $targets) {
        try {
            $result = Remove-CodexCleanupTarget -Target $target
            if ($result -eq 'Removed') {
                $removedCount++
                $line = '[已删除] ' + $target.Path
            }
            else {
                $missingCount++
                $line = '[不存在，跳过] ' + $target.Path
            }
            Write-Host $line -ForegroundColor Green
        }
        catch {
            $failureCount++
            $line = '[失败] ' + $target.Path + '：' + $_.Exception.Message
            Write-Host $line -ForegroundColor Red
        }
        $reportLines.Add($line)
    }
    $summary = '已删除 {0} 项，原本不存在 {1} 项，失败 {2} 项。' -f $removedCount, $missingCount, $failureCount
    Write-Host $summary
    $reportLines.Add($summary)
    if ($failureCount -gt 0) {
        Write-Host '有项目未清理成功。请根据失败信息处理；文件占用时可重启后再运行。不要强行修改系统目录权限。' -ForegroundColor Yellow
    }
    try {
        $reportPath = Join-Path -Path $PSScriptRoot -ChildPath 'Cleanup-Codex-result.txt'
        Set-Content -LiteralPath $reportPath -Value $reportLines.ToArray() -Encoding UTF8 -ErrorAction Stop
        Write-Host ('结果已保存：' + $reportPath)
    }
    catch {
        Write-Host ('结果文件保存失败，请保留窗口中的结果：' + $_.Exception.Message) -ForegroundColor Yellow
    }
    if ($failureCount -gt 0) { return 1 }
    return 0
}

# Dot-sourcing exposes the guarded helpers for isolated verification only.
if ($MyInvocation.InvocationName -ne '.') {
    try {
        exit (Invoke-CodexCleanup -Preview:$PreviewOnly)
    }
    catch {
        Write-Host ('已停止：' + $_.Exception.Message) -ForegroundColor Red
        exit 2
    }
}
