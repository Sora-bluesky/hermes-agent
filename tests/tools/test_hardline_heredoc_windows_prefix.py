"""Hardline command positions after heredoc redirects and quoted Windows paths."""

import pytest

from tools.approval import check_all_command_guards
from tools.approval_detection import detect_hardline_command, hardline_scan_incomplete


@pytest.mark.parametrize("command", [
    "<<EOF /sbin/shutdown -h now",
    "<<-EOF reboot",
    r'"C:\Program Files\Git\usr\bin\rm.exe" -rf /',
    pytest.param(r'"C:\Program Files\Git\usr\bin\rm.exe""" -rf /', id="empty-double-quote-suffix"),
    "reboot${x:+}",
    "2>/dev/null reboot",
    "sudo --use root /sbin/shutdown",
    "rm.exe -rf /",
])
def test_command_prefix_is_hardline(command, monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")
    assert detect_hardline_command(command)[0], command
    calls = []
    result = check_all_command_guards(
        command, "local", approval_callback=lambda *args, **kwargs: calls.append(args) or True,
    )
    assert result["approved"] is False
    assert result.get("hardline") is True
    assert calls == []


@pytest.mark.parametrize("command", [
    "/tmp/reboot}",
    "<<<reboot cat",
    "cat <<EOF",
    r'echo "C:\Program Files\Git\usr\bin\rm.exe" -rf /',
    pytest.param(r'''"C:\Program Files\Git\usr\bin\r'm.exe" -rf /''', id="literal-quote-in-filename"),
])
def test_arguments_and_other_executables_are_not_hardline(command):
    assert detect_hardline_command(command) == (False, None)
    assert not hardline_scan_incomplete(command)


@pytest.mark.parametrize("command", [
    '>"$(case x in x) true;; esac> /dev/null; printf /dev/null)" /sbin/reboot',
    '>"$(echo ${x:-${y:-"}"}} >/dev/null; printf /dev/null)" /sbin/reboot',
])
def test_redirect_target_keeps_following_reboot_hardline(command):
    assert detect_hardline_command(command)[0], command
