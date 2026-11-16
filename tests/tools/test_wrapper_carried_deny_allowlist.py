"""Wrapper-carried commands share deny, allowlist and hardline policy."""

import time

import pytest

from tools import approval, approval_context, approval_floors
from tools.approval_detection import detect_dangerous_command, detect_hardline_command


@pytest.fixture(autouse=True)
def approval_env(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")
    for name in ("HERMES_GATEWAY_SESSION", "HERMES_CRON_SESSION", "HERMES_EXEC_ASK"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(approval, "_YOLO_MODE_FROZEN", False)
    monkeypatch.setattr(approval_context, "_get_approval_config", lambda: {"mode": "manual"})
    monkeypatch.setattr(approval, "_permanent_set", lambda: set())
    token = approval_context.set_current_session_key(str(tmp_path))
    yield
    approval_context.reset_current_session_key(token)


@pytest.mark.parametrize("command", [
    "doas rm -rf /tmp/x", "doas -u root rm -rf /tmp/x",
    "xargs rm -rf /tmp/x", "xargs -0 -n 1 -I {} rm -rf /tmp/x",
    "watch -n1 rm -rf /tmp/x", "watch -n 1 rm -rf /tmp/x",
    "flock /tmp/l rm -rf /tmp/x", "flock -w 1 /tmp/l rm -rf /tmp/x",
    "systemd-run --scope rm -rf /tmp/x",
    "systemd-run --user -p X=Y --unit=N rm -rf /tmp/x",
    "su -c 'rm -rf /'", "runuser -l root -c 'rm -rf /'",
    "script -c 'rm -rf /' /dev/null", "flock /tmp/l -c 'rm -rf /'",
    "sudo rm -rf /tmp/x", "env X=1 rm -rf /tmp/x",
    "runuser -u root -- rm -rf /tmp/x", "runuser -u root rm -rf /tmp/x",
    "doas watch -n1 flock /tmp/l rm -rf /tmp/x",
    "xargs -n1 su -c 'rm -rf /tmp/x'",
])
def test_carried_deny_beats_yolo(command, monkeypatch):
    monkeypatch.setattr(approval_context, "_get_approval_config", lambda: {"deny": ["rm -rf*"]})
    monkeypatch.setattr(approval, "_YOLO_MODE_FROZEN", True)
    assert approval_floors._match_user_deny_rule(command) == "rm -rf*"
    prompts = []
    result = approval.check_all_command_guards(command, "local", approval_callback=prompts.append)
    assert result["approved"] is False
    # Catastrophic commands encounter the hardline floor before the user-deny floor.
    assert result.get("user_deny") or result.get("hardline")
    assert not prompts


@pytest.mark.parametrize("command", [
    "su -c 'rm -rf /tmp/x'", "doas rm -rf /tmp/x", "watch -n1 rm -rf /tmp/x",
    "watch doas rm -rf /tmp/x", "doas su -c 'rm -rf /tmp/x'",
    "xargs -n1 rm -rf /tmp/x", "flock /tmp/l rm -rf /tmp/x",
    "script -c 'rm -rf /tmp/x' out.txt", "systemd-run --scope rm -rf /tmp/x",
    "runuser -u root -- rm -rf /tmp/x",
])
def test_dangerous_payload_withdraws_wrapper_glob(command, monkeypatch):
    monkeypatch.setattr(approval, "_permanent_set", lambda: {
        "su *", "doas *", "watch *", "xargs *", "flock *", "script *", "systemd-run *", "runuser *", "ls",
    })
    assert not approval._command_matches_permanent_allowlist(command)
    prompts = []

    def deny(*args, **kwargs):
        prompts.append(args)
        return "deny"

    result = approval.check_all_command_guards(command, "local", approval_callback=deny)
    assert result["approved"] is False
    assert prompts


@pytest.mark.parametrize(("command", "patterns", "expected"), [
    ("watch ls", {"ls"}, False),
    ("watch ls", {"su *", "doas *", "watch *", "ls"}, True),
    ("doas ls", {"su *", "doas *", "watch *", "ls"}, True),
    ("su - alice", {"su *", "doas *", "watch *", "ls"}, True),
    ("doas rm -rf /", {"doas rm -rf /", "doas *"}, True),
    ("rm -rf /tmp/x", {"rm *"}, True),
])
def test_projection_only_withdraws_wrapper_globs(command, patterns, expected, monkeypatch):
    monkeypatch.setattr(approval, "_permanent_set", lambda: patterns)
    assert approval._command_matches_permanent_allowlist(command) is expected


@pytest.mark.parametrize("command", [
    "script -c 'rm -rf /' /dev/null", "flock /tmp/l -c 'rm -rf /'",
    "flock -c 'rm -rf /' /tmp/l", "runuser -u root -- reboot", "runuser -u root reboot",
    "flock -xc 'rm -rf /' /tmp/l", "flock -- /tmp/l reboot",
    "watch -n1 reboot", "systemd-run -t --scope reboot",
    "su -c 'rm -rf /'", "doas sh -c ':(){ :|:& };:'",
    "su root -c reboot", "runuser -l root -c reboot",
])
def test_carried_hardline(command):
    assert detect_hardline_command(command)[0]


@pytest.mark.parametrize("command", [
    "su -c ls", "script -c ls out.txt", 'flock -c "ls" /tmp/l',
    "su -c ls alice", "runuser -u root -- ls", "git log --oneline", "echo rm is a command",
    "doas -u reboot echo ok", "xargs -Ireboot echo ok", "watch -n1 echo reboot",
    "flock /tmp/reboot echo ok", "systemd-run -p reboot --unit=reboot echo ok",
    "script --log-out reboot -c ls", "runuser reboot",
    "flock /tmp/l echo -c reboot", "runuser -u root echo -c reboot",
])
def test_safe_commands_and_option_operands(command):
    assert not detect_hardline_command(command)[0]
    assert not detect_dangerous_command(command)[0]


@pytest.mark.parametrize("command", [
    "doas " * 300 + "ls",
    "; ".join(["watch -n1 ls"] * 300),
], ids=["nested", "repeated"])
@pytest.mark.parametrize(("guard", "expected"), [
    (approval_floors._match_user_deny_rule, None),
    (approval._command_matches_permanent_allowlist, False),
    (detect_hardline_command, (False, None)),
])
def test_wrapper_runtime_is_bounded(command, guard, expected, monkeypatch):
    monkeypatch.setattr(approval_context, "_get_approval_config", lambda: {"deny": ["rm -rf*"]})
    monkeypatch.setattr(approval, "_permanent_set", lambda: {"doas *", "watch *"})
    start = time.perf_counter()
    result = guard(command)
    assert time.perf_counter() - start < 1
    # A nested safe command still matches its wrapper glob; a compound command cannot.
    if guard is approval._command_matches_permanent_allowlist and command.startswith("doas"):
        expected = True
    assert result == expected


@pytest.mark.parametrize("options", [
    "-u reboot", "--unit reboot", "-p Nice=5", "--property Nice=5",
    "--description reboot", "--slice reboot.slice", "--expand-environment yes",
    "--service-type exec", "--uid root", "--gid root", "--nice 5",
    "--working-directory /tmp/reboot", "--root-directory /tmp/reboot",
    "-E X=reboot", "--setenv X=reboot", "--output journal",
    "--on-active 1", "--on-boot 1", "--on-startup 1", "--on-unit-active 1",
    "--on-unit-inactive 1", "--on-calendar daily", "--path-property PathExists=/tmp/x",
    "--socket-property ListenStream=12345", "--timer-property AccuracySec=1",
    "--job-mode replace", "--background yes", "-H reboot", "--host reboot",
    "-M reboot", "--machine reboot", "-C reboot", "--capsule reboot", "--json short",
    "--nice=5", "--on-boot=1", "-Creboot",
])
def test_systemd_value_operands(options, monkeypatch):
    monkeypatch.setattr(approval_context, "_get_approval_config", lambda: {"deny": ["rm -rf*"]})
    monkeypatch.setattr(approval, "_permanent_set", lambda: {"systemd-run *", "su *", "flock *", "script *"})
    prefix = f"systemd-run {options} "
    assert approval_floors._match_user_deny_rule(prefix + "rm -rf /tmp/x") == "rm -rf*"
    assert not approval._command_matches_permanent_allowlist(prefix + "rm -rf /tmp/x")
    assert detect_hardline_command(prefix + "reboot")[0]
    assert not detect_hardline_command(prefix + "echo ok")[0]


@pytest.mark.parametrize("option", [
    "--scope", "--slice-inherit", "-r", "--remain-after-exit", "--send-sighup",
    "-d", "--same-dir", "-R", "--same-root-dir", "-t", "--pty", "-T", "--pty-late",
    "-P", "--pipe", "-S", "--shell", "-q", "--quiet", "-v", "--verbose",
    "--on-clock-change", "--on-timezone-change", "--no-block", "--wait", "-G", "--collect",
    "--ignore-failure", "--user", "--system", "--no-ask-password",
])
def test_systemd_flags_leave_program_in_command_position(option):
    assert detect_hardline_command(f"systemd-run {option} /tmp/reboot echo ok")[0]


@pytest.mark.parametrize("command", [
    'su -c "su -c \\"su -c \'su -c reboot\'\\""',
    'su -c "su -c \\"su -c \'su -c ls\'\\""',
    'script -c "su -c \\"su -c \'su -c ls\'\\""',
    'flock /tmp/l -c "su -c \\"su -c \'su -c ls\'\\""',
])
def test_unread_carrier_payload_refuses_at_limit(command, monkeypatch):
    monkeypatch.setattr(approval, "_permanent_set", lambda: {"systemd-run *", "su *", "flock *", "script *"})
    monkeypatch.setattr(approval, "_YOLO_MODE_FROZEN", True)
    assert detect_hardline_command(command) == (True, "command parser limit exceeded")
    assert not approval._command_matches_permanent_allowlist(command)
    prompts = []
    result = approval.check_all_command_guards(command, "local", approval_callback=prompts.append)
    assert result["approved"] is False
    assert result["hardline"]
    assert not prompts


def test_fully_read_carrier_chain_at_limit_stays_allowlisted(monkeypatch):
    command = 'su -c "su -c \'su -c ls\'"'
    monkeypatch.setattr(approval, "_permanent_set", lambda: {"su *"})
    assert detect_hardline_command(command) == (False, None)
    assert approval._command_matches_permanent_allowlist(command)


@pytest.mark.parametrize("options", [
    "-s", "--shared", "-x", "-e", "--exclusive", "-u", "--unlock",
    "-n", "--nonblock", "--nb", "-o", "--close", "-F", "--no-fork", "--verbose",
    "-w 1", "--wait 1", "--timeout 1", "-E 1", "--conflict-exit-code 1",
    "-w1", "--timeout=1",
])
@pytest.mark.parametrize("template", ["flock {options} /tmp/l", "flock /tmp/l {options}"])
def test_flock_options_do_not_consume_lock_position(options, template, monkeypatch):
    monkeypatch.setattr(approval_context, "_get_approval_config", lambda: {"deny": ["rm -rf*"]})
    monkeypatch.setattr(approval, "_permanent_set", lambda: {"systemd-run *", "su *", "flock *", "script *"})
    prefix = template.format(options=options)
    command = prefix + " -c 'rm -rf /'"
    assert approval_floors._match_user_deny_rule(command) == "rm -rf*"
    assert not approval._command_matches_permanent_allowlist(command)
    assert detect_hardline_command(command)[0]
    assert not detect_hardline_command(prefix + " -c ls")[0]
    assert not detect_hardline_command(prefix + " echo -c reboot")[0]


@pytest.mark.parametrize("command", [
    "script > -c reboot", "script >-c reboot", "script 2> -c reboot",
    "script >> -c reboot", "script < -c reboot", "su > -c reboot",
    "runuser > -c reboot", "flock > -c /tmp/l echo ok",
    "script -c > reboot ls", "script --log-out > -c reboot -c ls",
])
def test_carrier_redirect_targets_are_not_options(command):
    assert not detect_hardline_command(command)[0]
    assert not detect_dangerous_command(command)[0]


@pytest.mark.parametrize("command", [
    "script > out -c reboot", "script '>' -c reboot", r"script \> -c reboot",
    "flock > out --exclusive /tmp/l -c reboot",
])
def test_carrier_redirects_preserve_real_command_options(command):
    assert detect_hardline_command(command)[0]


@pytest.mark.parametrize(("command", "denied", "allowlisted", "hardline"), [
    ("systemd-run --help reboot", False, True, False),
    ("watch --version reboot", False, True, False),
    ("systemd-run -u > out reboot echo ok", False, False, False),
    ("flock --exclusive /tmp/l rm -rf /tmp/x", True, False, False),
    ("script -c 2>&1 reboot", False, False, True),
    ("systemd-run -d /tmp/reboot echo ok", False, True, True),
    ("xargs --help reboot", False, True, True),
])
def test_wrapper_option_and_redirect_decisions(command, denied, allowlisted, hardline, monkeypatch):
    monkeypatch.setattr(approval_context, "_get_approval_config", lambda: {"deny": ["rm -rf*"]})
    monkeypatch.setattr(approval, "_permanent_set", lambda: {
        "script *", "watch *", "xargs *", "systemd-run *", "flock *", "runuser *",
    })
    assert approval_floors._match_user_deny_rule(command) == ("rm -rf*" if denied else None)
    assert approval._command_matches_permanent_allowlist(command) is allowlisted
    assert detect_hardline_command(command)[0] is hardline
    result = approval.check_all_command_guards(command, "local", approval_callback=lambda *a, **kw: "deny")
    assert result["approved"] is not (denied or hardline)


@pytest.mark.parametrize(("wrapper", "query"), [
    ("watch", "--help"), ("watch", "-h"), ("watch", "--version"), ("watch", "-v"),
    ("flock", "--help"), ("flock", "-h"), ("flock", "--version"), ("flock", "-V"),
    ("systemd-run", "--help"), ("systemd-run", "-h"), ("systemd-run", "--version"),
    ("runuser", "--help"), ("runuser", "-h"), ("runuser", "--version"), ("runuser", "-V"),
    ("script", "--help"), ("script", "-h"), ("script", "--version"), ("script", "-V"),
])
def test_wrapper_query_options_stop_execution(wrapper, query, monkeypatch):
    monkeypatch.setattr(approval_context, "_get_approval_config", lambda: {"deny": ["rm -rf*"]})
    monkeypatch.setattr(approval, "_permanent_set", lambda: {f"{wrapper} *"})
    commands = [f"{wrapper} {query} reboot", f"{wrapper} {query} rm -rf /tmp/x"]
    if wrapper in {"flock", "runuser", "script"}:
        commands.extend([f"{wrapper} {query} -c reboot", f"{wrapper} {query} -c 'rm -rf /tmp/x'"])
    for command in commands:
        assert not detect_hardline_command(command)[0], command
        assert approval_floors._match_user_deny_rule(command) is None, command
        assert approval._command_matches_permanent_allowlist(command), command


@pytest.mark.parametrize("redirect", ["> out", "2>&1", "0<&3", "&>out", "&>>out", "> out 2>&1"])
def test_redirects_before_wrapper_option_value(redirect):
    assert not detect_hardline_command(f"systemd-run -u {redirect} reboot echo ok")[0]
    assert detect_hardline_command(f"systemd-run -u {redirect} unit reboot")[0]


@pytest.mark.parametrize("redirect", ["2>&1", "0<&3", "&>out", "&>>out", ">|out"])
def test_carrier_redirect_operators_preserve_payload(redirect, monkeypatch):
    monkeypatch.setattr(approval_context, "_get_approval_config", lambda: {"deny": ["rm -rf*"]})
    assert detect_hardline_command(f"script -c {redirect} reboot")[0]
    assert approval_floors._match_user_deny_rule(f"script -c {redirect} 'rm -rf /tmp/x'") == "rm -rf*"
    assert not detect_hardline_command(f"script -c {redirect} ls")[0]


@pytest.mark.parametrize("command", [
    "script -O -h -c reboot", "script -creboot", "flock -w -V /tmp/l reboot",
    "runuser -u -h reboot", "watch -n -h reboot", "systemd-run -u -h reboot",
    "script -c ls & script -c reboot",
])
def test_query_words_as_operands_and_separate_commands_still_run(command):
    assert detect_hardline_command(command)[0]


@pytest.mark.parametrize("command", [
    "script -c ls & echo -c reboot", "script '>'& echo -c reboot",
    r"script \>& echo -c reboot",
])
def test_real_ampersand_separates_carrier_from_other_arguments(command):
    assert not detect_hardline_command(command)[0]
