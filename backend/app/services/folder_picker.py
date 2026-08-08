"""Server-side folder selection for the Browse… field.

Both mechanisms operate on the machine that RUNS the server — the folder the app
triages lives there, not on the browser's machine — so a browser <input> can't
reach it:

  • open_native_folder_dialog(): pops the OS-native "choose a folder" dialog on
    the server's own desktop. Windows only, via a short PowerShell script running
    FolderBrowserDialog in its OWN -STA process — no message-pump conflict with
    the Flask worker thread. Returns the chosen path, None when the user
    cancelled, or raises NativePickerUnavailable when there is no desktop to draw
    on (headless box, Linux vast.ai instance, PowerShell missing, timeout). The UI
    silently falls back to the in-app browser then.

  • list_subfolders(): read-only enumeration of a directory's immediate
    SUBFOLDERS (never files — nothing sensitive is ever streamed) plus the drive
    list for the roots view. Backs the in-app folder browser used from the LAN /
    tablet / Linux, where a server-side native dialog makes no sense.
"""
import logging
import os
import shutil
import string
import subprocess
import tempfile

logger = logging.getLogger(__name__)

# Give the human time to actually browse and pick before we give up on the
# native dialog and let the UI fall back to the in-app browser.
NATIVE_DIALOG_TIMEOUT = 180


class NativePickerUnavailable(RuntimeError):
    """The server has no usable native folder dialog (not Windows, no desktop,
    PowerShell missing, or the dialog timed out)."""


# PowerShell that shows a folder dialog and writes ONLY the chosen path to stdout
# (empty when cancelled). Runs under -STA (every shell dialog needs a
# single-threaded apartment); a TopMost owner form pulls it above the console.
# The initial path arrives via the LDS_PICKER_INITIAL env var, NOT stdin: this
# script is launched with `-File`, and stdin under `-Command -` would be consumed
# as the command text (which also stops ShowDialog from ever blocking). Output is
# forced to UTF-8 so paths with accents survive the trip back to Python.
#
# WHICH dialog, and why there are three tiers.
# `FolderBrowserDialog` is two completely different dialogs depending on the
# runtime underneath PowerShell, and that is the whole reason this file grew:
#   * .NET Framework 4.x (windows PowerShell 5.1) draws the XP-era tree — no
#     address bar, no Quick Access, and no way to paste a path. Users with a path
#     on the clipboard (the common case: a folder someone gave them, or one they
#     copied from Explorer) had to click their way down to it.
#   * .NET Core 3.0+ (pwsh 7) sets `AutoUpgradeEnabled` and draws the Vista-era
#     Common Item Dialog — address bar, Quick Access, pasteable path — for free.
# So: prefer the runtime that already does the right thing (see
# _powershell_exe), detect it by the presence of that property, and only fall
# back to hand-rolled COM interop when we are stuck on .NET Framework.
#
# Tier 1 (modern runtime): FolderBrowserDialog as-is. Microsoft-maintained, zero
#   interop surface.
# Tier 2 (.NET Framework): IFileOpenDialog + FOS_PICKFOLDERS through the interop
#   below — the same Common Item Dialog, driven by hand.
# Tier 3 (anything above threw): the legacy FolderBrowserDialog. A dated dialog
#   still picks folders; a broken one picks nothing, so the fallback is
#   unconditional and never surfaces the interop error to the user.
#
# The interop's method ORDER is load-bearing and is not guesswork: a COM vtable
# is positional, so one misplaced entry calls a different function with the wrong
# arguments. It is transcribed from .NET's own reference-source declaration
# (ndp/fx/src/WinForms/Managed/System/WinForms/FileDialog_Vista_Interop.cs) and
# stops at GetResult — the last slot we call. Do not reorder, and only append.
_PS_SCRIPT = r"""
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$initial = $env:LDS_PICKER_INITIAL
$mode = $env:LDS_PICKER_MODE            # '', 'auto', 'legacy', 'com'
if (-not $mode) { $mode = 'auto' }
$title = 'LoRA Dataset Studio - choose a folder'

Add-Type -AssemblyName System.Windows.Forms | Out-Null

# True when this runtime's FolderBrowserDialog IS already the Common Item Dialog.
function Test-ModernShellDialog {
  return [bool][System.Windows.Forms.FolderBrowserDialog].GetProperty('AutoUpgradeEnabled')
}

function Show-WinFormsDialog {
  $dlg = New-Object System.Windows.Forms.FolderBrowserDialog
  $dlg.Description = $title
  $dlg.ShowNewFolderButton = $true
  if ($initial -and (Test-Path -LiteralPath $initial -PathType Container)) {
    $dlg.SelectedPath = $initial
    # .NET Core also honours InitialDirectory, which is what actually opens the
    # Common Item Dialog AT that folder rather than merely preselecting it.
    $p = $dlg.GetType().GetProperty('InitialDirectory')
    if ($p) { $p.SetValue($dlg, $initial) }
  }
  $owner = New-Object System.Windows.Forms.Form
  $owner.TopMost = $true
  $owner.ShowInTaskbar = $false
  $owner.Opacity = 0
  try {
    if ($dlg.ShowDialog($owner) -eq [System.Windows.Forms.DialogResult]::OK) {
      return $dlg.SelectedPath
    }
    return $null
  } finally { $owner.Dispose(); $dlg.Dispose() }
}

$LdsPickerInterop = @'
using System;
using System.Runtime.InteropServices;

namespace LdsPicker {

  // Transcribed from .NET's FileDialog_Vista_Interop.cs. VTABLE ORDER IS THE
  // CONTRACT: entries are positional, so a reordered method silently calls its
  // neighbour. Truncated after GetResult (slot 18) because nothing below is
  // called; appending is safe, reordering is not.
  [ComImport, Guid("42f85136-db7e-439c-85f1-e4075d135fc8"),
   InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
  internal interface IFileDialog {
    [PreserveSig] int Show(IntPtr parent);
    void SetFileTypes(uint cFileTypes, IntPtr rgFilterSpec);
    void SetFileTypeIndex(uint iFileType);
    void GetFileTypeIndex(out uint piFileType);
    void Advise(IntPtr pfde, out uint pdwCookie);
    void Unadvise(uint dwCookie);
    void SetOptions(uint fos);
    void GetOptions(out uint pfos);
    void SetDefaultFolder(IShellItem psi);
    void SetFolder(IShellItem psi);
    void GetFolder(out IShellItem ppsi);
    void GetCurrentSelection(out IShellItem ppsi);
    void SetFileName([MarshalAs(UnmanagedType.LPWStr)] string pszName);
    void GetFileName([MarshalAs(UnmanagedType.LPWStr)] out string pszName);
    void SetTitle([MarshalAs(UnmanagedType.LPWStr)] string pszTitle);
    void SetOkButtonLabel([MarshalAs(UnmanagedType.LPWStr)] string pszText);
    void SetFileNameLabel([MarshalAs(UnmanagedType.LPWStr)] string pszLabel);
    void GetResult(out IShellItem ppsi);
  }

  [ComImport, Guid("43826d1e-e718-42ee-bc55-a1e261c37bfe"),
   InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
  internal interface IShellItem {
    void BindToHandler(IntPtr pbc, ref Guid bhid, ref Guid riid, out IntPtr ppv);
    void GetParent(out IShellItem ppsi);
    void GetDisplayName(uint sigdnName, [MarshalAs(UnmanagedType.LPWStr)] out string ppszName);
    void GetAttributes(uint sfgaoMask, out uint psfgaoAttribs);
    void Compare(IShellItem psi, uint hint, out int piOrder);
  }

  public static class FolderDialog {
    // shobjidl_core.h FILEOPENDIALOGOPTIONS
    private const uint FOS_PICKFOLDERS      = 0x00000020;
    private const uint FOS_FORCEFILESYSTEM  = 0x00000040;
    private const uint FOS_PATHMUSTEXIST    = 0x00000800;
    private const uint SIGDN_FILESYSPATH    = 0x80058000;
    // HRESULT_FROM_WIN32(ERROR_CANCELLED): the user closed the dialog. It is a
    // normal outcome, not a failure, and must NOT trigger the legacy fallback —
    // re-showing a second dialog after a deliberate cancel is a trap.
    private const int  ERROR_CANCELLED_HR   = unchecked((int)0x800704C7);
    private static readonly Guid CLSID_FileOpenDialog =
      new Guid("DC1C5A9C-E88A-4dde-A5A1-60F82A20AEF7");
    private static readonly Guid IID_IShellItem =
      new Guid("43826d1e-e718-42ee-bc55-a1e261c37bfe");

    [DllImport("shell32.dll", CharSet = CharSet.Unicode, PreserveSig = false)]
    private static extern void SHCreateItemFromParsingName(
      [MarshalAs(UnmanagedType.LPWStr)] string pszPath, IntPtr pbc, ref Guid riid,
      [MarshalAs(UnmanagedType.Interface)] out object ppv);

    private static IFileDialog NewDialog() {
      Type t = Type.GetTypeFromCLSID(CLSID_FileOpenDialog, true);
      return (IFileDialog)Activator.CreateInstance(t);
    }

    private static void ApplyFolderMode(IFileDialog dlg, string title, string initial) {
      uint opts;
      dlg.GetOptions(out opts);
      dlg.SetOptions(opts | FOS_PICKFOLDERS | FOS_FORCEFILESYSTEM | FOS_PATHMUSTEXIST);
      if (!string.IsNullOrEmpty(title)) dlg.SetTitle(title);
      if (!string.IsNullOrEmpty(initial)) {
        try {
          Guid riid = IID_IShellItem;
          object item;
          SHCreateItemFromParsingName(initial, IntPtr.Zero, ref riid, out item);
          // SetFolder, not SetDefaultFolder: the caller passes the folder the
          // user is already working in, which should win over the shell's
          // remembered last-used location.
          dlg.SetFolder((IShellItem)item);
        } catch { /* a stale initial path must not stop the dialog opening */ }
      }
    }

    // Returns the chosen path, or null when the user cancelled.
    public static string Pick(string initial, string title) {
      IFileDialog dlg = NewDialog();
      try {
        ApplyFolderMode(dlg, title, initial);
        int hr = dlg.Show(IntPtr.Zero);
        if (hr == ERROR_CANCELLED_HR) return null;
        if (hr != 0) Marshal.ThrowExceptionForHR(hr);
        IShellItem res;
        dlg.GetResult(out res);
        string path;
        res.GetDisplayName(SIGDN_FILESYSPATH, out path);
        Marshal.ReleaseComObject(res);
        return path;
      } finally { Marshal.ReleaseComObject(dlg); }
    }

    // Exercises every vtable slot this file depends on WITHOUT showing a window,
    // so the interop can be proven on a machine with nobody in front of it.
    // A shifted vtable cannot survive this: the options round-trip carries a
    // distinctive value, and the folder round-trip goes out as an interface
    // pointer and comes back as a filesystem string.
    public static string SelfTest(string probeFolder) {
      IFileDialog dlg = NewDialog();
      try {
        uint before, after;
        dlg.GetOptions(out before);
        dlg.SetOptions(before | FOS_PICKFOLDERS | FOS_FORCEFILESYSTEM);
        dlg.GetOptions(out after);
        bool flags = (after & FOS_PICKFOLDERS) != 0 && (after & FOS_FORCEFILESYSTEM) != 0;
        dlg.SetTitle("LoRA Dataset Studio self-test");
        string roundTrip = "(skipped)";
        if (!string.IsNullOrEmpty(probeFolder)) {
          Guid riid = IID_IShellItem;
          object item;
          SHCreateItemFromParsingName(probeFolder, IntPtr.Zero, ref riid, out item);
          dlg.SetFolder((IShellItem)item);
          IShellItem back;
          dlg.GetFolder(out back);
          back.GetDisplayName(SIGDN_FILESYSPATH, out roundTrip);
          Marshal.ReleaseComObject(back);
          Marshal.ReleaseComObject(item);
        }
        return "options_before=0x" + before.ToString("X")
             + " options_after=0x" + after.ToString("X")
             + " pickfolders_set=" + flags
             + " folder_roundtrip=" + roundTrip;
      } finally { Marshal.ReleaseComObject(dlg); }
    }
  }
}
'@

function Show-ComDialog {
  if (-not ('LdsPicker.FolderDialog' -as [type])) {
    Add-Type -TypeDefinition $LdsPickerInterop -Language CSharp | Out-Null
  }
  return [LdsPicker.FolderDialog]::Pick($initial, $title)
}

# Self-test hook: prove the interop end to end without a human at the screen.
if ($env:LDS_PICKER_SELFTEST) {
  if (-not ('LdsPicker.FolderDialog' -as [type])) {
    Add-Type -TypeDefinition $LdsPickerInterop -Language CSharp | Out-Null
  }
  [Console]::Out.Write([LdsPicker.FolderDialog]::SelfTest($initial))
  exit 0
}

$path = $null
$useCom = ($mode -eq 'com') -or ($mode -eq 'auto' -and -not (Test-ModernShellDialog))
if ($mode -eq 'legacy') {
  $path = Show-WinFormsDialog
} elseif ($useCom) {
  try {
    $path = Show-ComDialog
  } catch {
    # Constrained language mode, no C# compiler, a locked-down COM policy: the
    # user still gets A folder picker. The reason goes to stderr, which Python
    # only reads when the whole script fails.
    [Console]::Error.Write('IFileOpenDialog unavailable, falling back: ' + $_.Exception.Message)
    $path = Show-WinFormsDialog
  }
} else {
  $path = Show-WinFormsDialog
}
if ($path) { [Console]::Out.Write($path) }
"""


def native_dialog_available():
    """Cheap pre-check: Windows + a PowerShell on PATH. The real proof is only
    known once we try (there may be no interactive desktop), but this rejects the
    obvious non-Windows / vast.ai case without spawning a process."""
    return os.name == 'nt' and bool(_powershell_exe())


def _powershell_exe():
    """The PowerShell to run the picker under — pwsh FIRST when it exists.

    Not a stylistic preference: the host decides which dialog the user gets.
    pwsh 7 runs on .NET Core, whose FolderBrowserDialog is the modern Common Item
    Dialog (address bar, Quick Access, pasteable path); powershell.exe 5.1 runs on
    .NET Framework, whose FolderBrowserDialog is the XP-era tree. Picking
    powershell.exe first — as this did — handed the dated dialog to every machine
    that had both installed, while the good one sat one PATH entry away.

    Both are launched with -STA explicitly (pwsh defaults to MTA), so the
    apartment is the same either way.
    """
    return shutil.which('pwsh') or shutil.which('powershell')


def open_native_folder_dialog(initial=None):
    """Show the server-side native folder dialog. Returns the selected path, or
    None if the user cancelled. Raises NativePickerUnavailable when there is no
    native dialog to show (see module docstring)."""
    exe = _powershell_exe()
    if os.name != 'nt' or not exe:
        raise NativePickerUnavailable('native folder dialog is Windows-only')
    # Run the script from a temp -File (NOT `-Command -`): piping the script over
    # stdin makes PowerShell treat all of stdin as the command and ShowDialog
    # returns immediately without ever painting. The initial path rides in via an
    # env var; stderr is captured as bytes and decoded leniently so a localized
    # (non-UTF-8) PowerShell error never crashes the reader thread.
    env = dict(os.environ, LDS_PICKER_INITIAL=(initial or ''))
    tmp = tempfile.NamedTemporaryFile(
        'w', suffix='.ps1', delete=False, encoding='utf-8')
    try:
        tmp.write(_PS_SCRIPT)
        tmp.close()
        proc = subprocess.run(
            [exe, '-NoProfile', '-STA', '-ExecutionPolicy', 'Bypass',
             '-File', tmp.name],
            env=env, capture_output=True, timeout=NATIVE_DIALOG_TIMEOUT)
    except subprocess.TimeoutExpired as e:
        raise NativePickerUnavailable('the folder dialog timed out') from e
    except OSError as e:  # PowerShell vanished between the which() and the run
        raise NativePickerUnavailable(f'could not launch the dialog: {e}') from e
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
    if proc.returncode != 0:
        # A non-zero exit here means the script itself blew up (e.g. no desktop /
        # WinForms unavailable on a service session) — treat as unavailable so the
        # UI falls back rather than surfacing a raw PowerShell trace.
        logger.info('native folder dialog unavailable: %s',
                    (proc.stderr or b'').decode('utf-8', 'replace').strip()[:200])
        raise NativePickerUnavailable('the native folder dialog is unavailable')
    path = (proc.stdout or b'').decode('utf-8', 'replace').strip()
    return path or None


def list_drives():
    """Root entries for the browser: drive letters on Windows, '/' on POSIX."""
    if os.name == 'nt':
        drives = []
        for letter in string.ascii_uppercase:
            root = f'{letter}:\\'
            if os.path.isdir(root):
                drives.append({'name': root, 'path': root})
        return drives
    return [{'name': '/', 'path': '/'}]


def list_subfolders(path=None):
    """Read-only listing for the in-app folder browser.

    path falsy  -> the roots view (drives / '/'), no parent.
    path given  -> that directory's immediate SUBFOLDERS only (never files),
                   with the parent for an "up" step. Path is normalized (abspath
                   collapses any '..'); unreadable entries are skipped, not fatal.
    """
    if not path or not str(path).strip():
        return {'path': None, 'parent': None, 'is_root': True,
                'entries': list_drives()}

    p = os.path.abspath(os.path.expanduser(str(path).strip()))
    if not os.path.isdir(p):
        raise ValueError('That folder does not exist.')

    entries = []
    with os.scandir(p) as it:
        for e in it:
            try:
                if e.is_dir():
                    entries.append({'name': e.name, 'path': e.path})
            except OSError:
                continue  # a junction/symlink we can't stat — skip, don't fail
    entries.sort(key=lambda d: d['name'].lower())

    parent = os.path.dirname(p)
    # At a filesystem root, dirname() is a no-op (returns p); surface the roots
    # view as the parent instead so "up" from C:\ lands on the drive list.
    at_root = parent == p
    return {'path': p, 'parent': None if at_root else parent,
            'is_root': False, 'entries': entries}
