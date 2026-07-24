"""
Tests for Home Assistant WebSocket gateway network-failure recovery.

Regression coverage for #67470 — the HA adapter could go silently deaf after
transient network failures:
1. A raised ``ws_connect()`` leaked the just-created ``aiohttp.ClientSession``.
2. Teardown awaits (``ws.close()`` / ``session.close()``) had no timeout and
   could block forever on a wedged CLOSE-WAIT socket.
3. The auth-handshake ``receive_json()`` calls had no timeout, so a server
   that accepted the socket but never responded froze ``_ws_connect``.
4. Nothing detected a wedged ``_listen_loop`` task — the gateway stayed
   "running" but silently stopped processing events.
"""

import asyncio
import gc
import time
import weakref
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.homeassistant import adapter as ha_adapter
from plugins.platforms.homeassistant.adapter import HomeAssistantAdapter


def _make_adapter(**extra) -> HomeAssistantAdapter:
    config = PlatformConfig(enabled=True, token="tok", extra=extra)
    return HomeAssistantAdapter(config)


async def _hang_forever(*_args, **_kwargs):
    await asyncio.Event().wait()


async def _await_tracked_teardown(tasks, *, timeout: float = 2) -> None:
    """Deterministically wait for previously snapshotted background
    teardown task(s) (from ``adapter._teardown_tasks``) to fully finish.

    ``_cleanup_ws()`` / ``_cancel_safe_close()`` / ``_full_teardown()``
    shield their close from a caller's cancellation by running it as a
    task tracked in ``self._teardown_tasks``; a *second* cancellation only
    detaches the caller from that task, it does not stop the task itself.
    A signal (``asyncio.Event``) set partway through that task's own body
    only proves ONE step ran -- it does not prove every *later* step in
    the same shielded unit has also run yet on this scheduler. That gap is
    exactly what made
    ``test_cleanup_ws_closes_session_when_cancelled_during_ws_close``
    flaky on Linux CI (failed: session.close awaited 0 times) while
    passing on Windows by scheduling coincidence -- it waited on the
    event from the WS close (step 1) and then immediately asserted on the
    session close (step 2) of the same shielded ``_close_both()`` task.

    Awaiting the task object(s) directly has no such gap: ``gather()``
    only returns once each task's whole coroutine body -- every step in
    it, including nested shielded sub-tasks -- is actually done. Callers
    must snapshot ``tuple(adapter._teardown_tasks)`` at the point where
    the tracked task is known to already exist (e.g. right after an
    ``asyncio.Event`` set from inside that task's first step fires), then
    pass the snapshot here after the caller-side cancellation.
    """
    assert tasks, "expected a background teardown task to already be tracked"
    await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=timeout)


# ---------------------------------------------------------------------------
# Defect 1: session leak on failed connect
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ws_connect_failed_connect_closes_local_session():
    """A raised ws_connect() must close the just-created local session
    instead of leaking it via a self._session that was assigned before the
    connect attempt (#67470)."""
    adapter = _make_adapter()

    mock_session = MagicMock()
    mock_session.closed = False
    mock_session.close = AsyncMock()
    mock_session.ws_connect = AsyncMock(side_effect=ConnectionError("refused"))

    with patch("plugins.platforms.homeassistant.adapter.aiohttp") as mock_aiohttp:
        mock_aiohttp.ClientTimeout = lambda total: total
        mock_aiohttp.ClientSession = MagicMock(return_value=mock_session)

        with pytest.raises(ConnectionError):
            await adapter._ws_connect()

    mock_session.close.assert_awaited_once()
    # The failed local session must never have been wired onto the adapter.
    assert adapter._session is None
    assert adapter._ws is None


# ---------------------------------------------------------------------------
# Defect 2: unbounded teardown awaits
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cleanup_ws_bounds_hanging_close_and_nulls_both(monkeypatch):
    """A wedged ws.close() must not block session.close() from running, and
    both attributes must be nulled regardless (#67470)."""
    adapter = _make_adapter()
    monkeypatch.setattr(ha_adapter, "_DRAIN_TIMEOUT", 0.05, raising=False)

    hung_ws = MagicMock()
    hung_ws.closed = False
    hung_ws.close = AsyncMock(side_effect=_hang_forever)

    session = MagicMock()
    session.closed = False
    session.close = AsyncMock()

    adapter._ws = hung_ws
    adapter._session = session

    await asyncio.wait_for(adapter._cleanup_ws(), timeout=2)

    assert adapter._ws is None
    assert adapter._session is None
    # The session close must still have run despite the ws close hanging.
    session.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_disconnect_completes_within_bounds_when_closes_hang(monkeypatch):
    """disconnect() must complete even when every underlying close() hangs."""
    adapter = _make_adapter()
    monkeypatch.setattr(ha_adapter, "_DRAIN_TIMEOUT", 0.05, raising=False)

    ws = MagicMock()
    ws.closed = False
    ws.close = AsyncMock(side_effect=_hang_forever)

    session = MagicMock()
    session.closed = False
    session.close = AsyncMock(side_effect=_hang_forever)

    rest_session = MagicMock()
    rest_session.closed = False
    rest_session.close = AsyncMock(side_effect=_hang_forever)

    adapter._ws = ws
    adapter._session = session
    adapter._rest_session = rest_session
    adapter._running = True

    async def _noop():
        return

    adapter._listen_task = asyncio.ensure_future(_noop())
    adapter._watchdog_task = asyncio.ensure_future(_noop())
    await asyncio.sleep(0)  # let the no-op tasks finish before disconnect() awaits them

    await asyncio.wait_for(adapter.disconnect(), timeout=2)

    assert adapter._ws is None
    assert adapter._session is None
    assert adapter._rest_session is None
    assert adapter._running is False


# ---------------------------------------------------------------------------
# Defect 3: unbounded auth handshake reads
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ws_connect_bounds_hanging_auth_handshake(monkeypatch):
    """A server that accepts the socket but never responds to
    receive_json() must not freeze _ws_connect() forever (#67470)."""
    adapter = _make_adapter()
    monkeypatch.setattr(ha_adapter, "_HANDSHAKE_TIMEOUT", 0.05, raising=False)

    mock_ws = MagicMock()
    mock_ws.closed = False
    mock_ws.receive_json = AsyncMock(side_effect=_hang_forever)
    mock_ws.send_json = AsyncMock()
    mock_ws.close = AsyncMock()

    mock_session = MagicMock()
    mock_session.closed = False
    mock_session.close = AsyncMock()
    mock_session.ws_connect = AsyncMock(return_value=mock_ws)

    with patch("plugins.platforms.homeassistant.adapter.aiohttp") as mock_aiohttp:
        mock_aiohttp.ClientTimeout = lambda total: total
        mock_aiohttp.ClientSession = MagicMock(return_value=mock_session)

        result = await asyncio.wait_for(adapter._ws_connect(), timeout=2)

    assert result is False
    mock_ws.close.assert_awaited_once()
    mock_session.close.assert_awaited_once()
    assert adapter._ws is None
    assert adapter._session is None


# ---------------------------------------------------------------------------
# Defect 4: cause-agnostic watchdog over _listen_loop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_watchdog_respawns_wedged_listen_task(monkeypatch):
    """If _last_progress goes stale past _LISTEN_STUCK_TIMEOUT while running,
    the watchdog must cancel the stuck listen task, force a cleanup, and
    respawn a new _listen_loop task (#67470)."""
    adapter = _make_adapter()
    monkeypatch.setattr(ha_adapter, "_WATCHDOG_INTERVAL", 0.01, raising=False)
    monkeypatch.setattr(ha_adapter, "_LISTEN_STUCK_TIMEOUT", 0.01, raising=False)

    respawn_calls = []

    async def _stub_listen_loop():
        respawn_calls.append(1)
        await asyncio.Event().wait()

    adapter._listen_loop = _stub_listen_loop  # type: ignore[method-assign]
    adapter._cleanup_ws = AsyncMock()

    adapter._running = True
    stuck_task = asyncio.ensure_future(_hang_forever())
    adapter._listen_task = stuck_task
    adapter._last_progress = time.monotonic() - 10  # already stale

    watchdog_task = asyncio.ensure_future(adapter._watchdog_loop())

    for _ in range(100):
        await asyncio.sleep(0.01)
        if respawn_calls and adapter._listen_task is not stuck_task:
            break

    assert respawn_calls, "watchdog must respawn a new _listen_loop task"
    assert adapter._listen_task is not None
    assert adapter._listen_task is not stuck_task
    assert stuck_task.cancelled()
    adapter._cleanup_ws.assert_awaited()

    # Clean up outstanding tasks.
    adapter._running = False
    for t in (watchdog_task, adapter._listen_task):
        t.cancel()
        try:
            await t
        except (asyncio.CancelledError, Exception):
            pass


@pytest.mark.asyncio
async def test_watchdog_stops_when_running_goes_false(monkeypatch):
    """The watchdog loop must exit cleanly once self._running is False."""
    adapter = _make_adapter()
    monkeypatch.setattr(ha_adapter, "_WATCHDOG_INTERVAL", 0.01, raising=False)
    adapter._running = True

    watchdog_task = asyncio.ensure_future(adapter._watchdog_loop())
    await asyncio.sleep(0.03)
    adapter._running = False

    await asyncio.wait_for(watchdog_task, timeout=2)
    assert watchdog_task.done()
    assert not watchdog_task.cancelled()


# ---------------------------------------------------------------------------
# Review follow-ups (#67470): handshake exception cleanup, quiet-vs-wedged
# ping probe, and bounded cancellation of uncancellable tasks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ws_connect_cleans_up_when_handshake_send_raises():
    """A send_json() that raises mid-handshake must tear the connection down
    inside _ws_connect() instead of leaking it to a later loop pass."""
    adapter = _make_adapter()

    mock_ws = MagicMock()
    mock_ws.closed = False
    mock_ws.receive_json = AsyncMock(return_value={"type": "auth_required"})
    mock_ws.send_json = AsyncMock(side_effect=ConnectionResetError("peer gone"))
    mock_ws.close = AsyncMock()

    mock_session = MagicMock()
    mock_session.closed = False
    mock_session.close = AsyncMock()
    mock_session.ws_connect = AsyncMock(return_value=mock_ws)

    with patch("plugins.platforms.homeassistant.adapter.aiohttp") as mock_aiohttp:
        mock_aiohttp.ClientTimeout = lambda total: total
        mock_aiohttp.ClientSession = MagicMock(return_value=mock_session)

        result = await asyncio.wait_for(adapter._ws_connect(), timeout=2)

    assert result is False
    mock_ws.close.assert_awaited_once()
    mock_session.close.assert_awaited_once()
    assert adapter._ws is None
    assert adapter._session is None


@pytest.mark.asyncio
async def test_watchdog_ping_probe_spares_quiet_but_healthy_listener(monkeypatch):
    """aiohttp answers heartbeat PINGs internally, so a healthy-but-quiet HA
    produces no reader frames. The watchdog's HA-protocol ping must detect the
    live listener (pong bumps _last_progress) and skip the respawn."""
    adapter = _make_adapter()
    monkeypatch.setattr(ha_adapter, "_WATCHDOG_INTERVAL", 0.01, raising=False)
    monkeypatch.setattr(ha_adapter, "_LISTEN_STUCK_TIMEOUT", 0.01, raising=False)
    monkeypatch.setattr(ha_adapter, "_PING_GRACE", 0.01, raising=False)

    async def _pong_arrives(payload):
        # Simulate the reader receiving the pong frame.
        adapter._last_progress = time.monotonic()

    live_ws = MagicMock()
    live_ws.closed = False
    live_ws.send_json = AsyncMock(side_effect=_pong_arrives)

    adapter._ws = live_ws
    adapter._running = True
    listen_task = asyncio.ensure_future(_hang_forever())
    adapter._listen_task = listen_task
    adapter._last_progress = time.monotonic() - 10  # stale by progress alone

    watchdog_task = asyncio.ensure_future(adapter._watchdog_loop())
    await asyncio.sleep(0.2)

    assert adapter._listen_task is listen_task, \
        "healthy-but-quiet listener must not be respawned"
    assert not listen_task.cancelled()
    live_ws.send_json.assert_awaited()

    adapter._running = False
    for t in (watchdog_task, listen_task):
        t.cancel()
        try:
            await t
        except (asyncio.CancelledError, Exception):
            pass


@pytest.mark.asyncio
async def test_watchdog_ping_probe_failure_respawns(monkeypatch):
    """A ping that cannot even be sent means the socket is wedged — the
    watchdog must proceed with the cancel-and-respawn recovery."""
    adapter = _make_adapter()
    monkeypatch.setattr(ha_adapter, "_WATCHDOG_INTERVAL", 0.01, raising=False)
    monkeypatch.setattr(ha_adapter, "_LISTEN_STUCK_TIMEOUT", 0.01, raising=False)
    monkeypatch.setattr(ha_adapter, "_PING_GRACE", 0.01, raising=False)

    dead_ws = MagicMock()
    dead_ws.closed = False
    dead_ws.send_json = AsyncMock(side_effect=ConnectionResetError("wedged"))

    respawn_calls = []

    async def _stub_listen_loop():
        respawn_calls.append(1)
        await asyncio.Event().wait()

    adapter._listen_loop = _stub_listen_loop  # type: ignore[method-assign]
    adapter._cleanup_ws = AsyncMock()
    adapter._ws = dead_ws
    adapter._running = True
    stuck_task = asyncio.ensure_future(_hang_forever())
    adapter._listen_task = stuck_task
    adapter._last_progress = time.monotonic() - 10

    watchdog_task = asyncio.ensure_future(adapter._watchdog_loop())

    for _ in range(100):
        await asyncio.sleep(0.01)
        if respawn_calls and adapter._listen_task is not stuck_task:
            break

    assert respawn_calls, "watchdog must respawn after a failed ping probe"
    assert stuck_task.cancelled()

    adapter._running = False
    for t in (watchdog_task, adapter._listen_task):
        t.cancel()
        try:
            await t
        except (asyncio.CancelledError, Exception):
            pass


@pytest.mark.asyncio
async def test_cancel_task_bounded_abandons_uncancellable_task(monkeypatch):
    """A task that swallows CancelledError must not hang the watchdog or
    disconnect(): _cancel_task_bounded gives up after _DRAIN_TIMEOUT."""
    adapter = _make_adapter()
    monkeypatch.setattr(ha_adapter, "_DRAIN_TIMEOUT", 0.05, raising=False)

    started = asyncio.Event()
    stop = asyncio.Event()

    async def _ignores_cancel():
        while not stop.is_set():
            try:
                started.set()
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                continue  # rude task refuses to die

    zombie = asyncio.ensure_future(_ignores_cancel())
    await started.wait()

    await asyncio.wait_for(
        adapter._cancel_task_bounded(zombie, "zombie"), timeout=2
    )

    assert not zombie.done(), "the zombie survives; the point is WE returned"
    # Final teardown for test hygiene: flip the stop flag, then nudge the
    # sleep with one more cancel so the loop re-checks it and exits normally.
    # (No wait_for on the zombie — wait_for's timeout path awaits cancellation
    # completing, the exact hang the production helper avoids.)
    stop.set()
    zombie.cancel()
    await asyncio.wait({zombie}, timeout=1)


# ---------------------------------------------------------------------------
# Review follow-up (#67470, egilewski): CancelledError bypasses the
# `except Exception` cleanup in _ws_connect(), leaking the freshly created
# session before self._session is ever assigned.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ws_connect_cancellation_closes_local_session():
    """A cancellation landing inside session.ws_connect() -- e.g. the
    gateway's outer per-platform connect deadline
    (asyncio.wait_for(adapter.connect(...), timeout=...) in
    gateway/run.py's _connect_adapter_with_timeout), or disconnect()/the
    watchdog cancelling an in-flight reconnect attempt -- must still close
    the freshly created ClientSession instead of leaking it.
    asyncio.CancelledError derives from BaseException, not Exception, so a
    plain `except Exception` never sees it and the close is skipped
    entirely (egilewski's probe: session.close observed awaited 0 times)."""
    adapter = _make_adapter()

    mock_session = MagicMock()
    mock_session.closed = False
    mock_session.close = AsyncMock()
    mock_session.ws_connect = AsyncMock(side_effect=_hang_forever)

    with patch("plugins.platforms.homeassistant.adapter.aiohttp") as mock_aiohttp:
        mock_aiohttp.ClientTimeout = lambda total: total
        mock_aiohttp.ClientSession = MagicMock(return_value=mock_session)

        with pytest.raises(asyncio.TimeoutError):
            # Mirrors gateway/run.py's outer connect deadline: it wraps the
            # whole connect() (which calls _ws_connect() first) in
            # asyncio.wait_for(..., timeout=...), cancelling it once the
            # deadline passes while ws_connect() is still hung.
            await asyncio.wait_for(adapter._ws_connect(), timeout=0.05)

    # No race here despite the close being shielded internally: there is
    # only ONE cancellation in this test (wait_for's own timeout), and
    # wait_for's cancel-then-wait machinery (_cancel_and_wait) does not
    # raise TimeoutError to us until the cancelled task is fully done --
    # which, since nothing cancels it a second time, requires
    # _cancel_safe_close()'s `await asyncio.shield(task)` to have returned
    # normally, i.e. the close already fully completed. Provable from
    # asyncio.wait_for's documented semantics, not from timing.
    mock_session.close.assert_awaited_once()
    assert adapter._session is None
    assert adapter._ws is None


@pytest.mark.asyncio
async def test_ws_connect_close_survives_second_cancellation_during_teardown():
    """A naive `except CancelledError: await self._bounded_close(...);
    raise` is not enough: this codebase has a real second cancellation
    source that can race in while that close is still running (the
    watchdog's wedged-listener respawn and disconnect() can both cancel the
    same listen task -- adapter.py's _cancel_task_bounded call sites at
    disconnect() and _watchdog_loop()). If that second cancellation
    interrupts the close before it finishes, the session leaks exactly like
    the original bug, just one frame deeper. The close must still complete
    (detached, tracked in self._teardown_tasks) even under this race."""
    adapter = _make_adapter()

    close_started = asyncio.Event()
    close_completed = asyncio.Event()
    mock_session = MagicMock()
    mock_session.closed = False

    async def _slow_close():
        close_started.set()
        await asyncio.sleep(0.05)
        close_completed.set()

    mock_session.close = _slow_close
    mock_session.ws_connect = AsyncMock(side_effect=_hang_forever)

    with patch("plugins.platforms.homeassistant.adapter.aiohttp") as mock_aiohttp:
        mock_aiohttp.ClientTimeout = lambda total: total
        mock_aiohttp.ClientSession = MagicMock(return_value=mock_session)

        task = asyncio.ensure_future(adapter._ws_connect())
        await asyncio.sleep(0)
        task.cancel()  # cancel #1: lands inside session.ws_connect()
        # Bounded: on the pre-fix code this never fires (the close is
        # skipped entirely), so an unbounded wait here would hang the
        # suite instead of failing fast.
        await asyncio.wait_for(close_started.wait(), timeout=2)
        # Snapshot the shielded close's tracked task now (it must already
        # exist -- close_started is set from inside it) so we can wait for
        # the WHOLE thing to finish afterward, deterministically.
        tracked = tuple(adapter._teardown_tasks)
        task.cancel()  # cancel #2: races in while that close is in flight
        with pytest.raises(asyncio.CancelledError):
            await task

    # The second cancellation only detaches `task` (the caller) from the
    # close; the close itself keeps running as the tracked background task
    # snapshotted above. See _await_tracked_teardown for why this must be
    # awaited directly rather than close_completed (Linux CI determinism).
    await _await_tracked_teardown(tracked)
    assert close_completed.is_set(), "the session close must have completed"
    assert adapter._session is None
    assert adapter._ws is None


@pytest.mark.asyncio
async def test_cleanup_ws_close_survives_second_cancellation_during_teardown():
    """_cleanup_ws() clears self._ws/self._session to None before awaiting
    their close -- reached from _ws_connect()'s handshake CancelledError
    handler and the _listen_loop reconnect ladder. If a second cancellation
    lands on the caller while that close is still running, the close must
    still complete instead of being interrupted mid-flight, even though the
    fields are already unreachable by that point (#67470 review,
    egilewski: this is the same leak class as _ws_connect()'s
    pre-assignment session, reached one level deeper)."""
    adapter = _make_adapter()

    ws = MagicMock()
    ws.closed = False
    ws.close = AsyncMock()

    session = MagicMock()
    session.closed = False
    close_started = asyncio.Event()
    close_completed = asyncio.Event()

    async def _slow_close():
        close_started.set()
        await asyncio.sleep(0.05)
        close_completed.set()

    session.close = _slow_close
    adapter._ws = ws
    adapter._session = session

    async def _handshake_cancelled_mid_cleanup():
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await adapter._cleanup_ws()
            raise

    task = asyncio.ensure_future(_handshake_cancelled_mid_cleanup())
    await asyncio.sleep(0)
    task.cancel()  # cancel #1: enters the CancelledError handler
    # Bounded: on the pre-fix code this never fires (the close is skipped
    # entirely), so an unbounded wait here would hang the suite instead of
    # failing fast.
    await asyncio.wait_for(close_started.wait(), timeout=2)
    # Snapshot the shielded close's tracked task now (it must already
    # exist -- close_started is set from inside it) so we can wait for the
    # WHOLE thing to finish afterward, deterministically.
    tracked = tuple(adapter._teardown_tasks)
    task.cancel()  # cancel #2: races in while _cleanup_ws() awaits the close
    with pytest.raises(asyncio.CancelledError):
        await task

    # The second cancellation only detaches `task` (the caller) from the
    # close; the close itself keeps running as the tracked background task
    # snapshotted above. See _await_tracked_teardown for why this must be
    # awaited directly rather than close_completed (Linux CI determinism).
    await _await_tracked_teardown(tracked)
    assert close_completed.is_set(), "the session close must have completed"
    assert adapter._ws is None
    assert adapter._session is None


@pytest.mark.asyncio
async def test_cleanup_ws_closes_session_when_cancelled_during_ws_close():
    """A cancellation racing in while _cleanup_ws()'s WebSocket close is
    still running must not skip the session close entirely (#67470 review
    round 2, egilewski). Running the WS close and session close as two
    separately shielded steps meant that second cancellation propagated out
    of the WS close's shielded await, so _cleanup_ws() returned before ever
    reaching the code that even attempts the session close -- leaving
    self._session non-None with nothing left running to close it. Both
    closes must run as one shielded unit so the whole thing keeps going."""
    adapter = _make_adapter()

    ws = MagicMock()
    ws.closed = False
    ws_close_started = asyncio.Event()
    ws_close_completed = asyncio.Event()

    async def _slow_ws_close():
        ws_close_started.set()
        await asyncio.sleep(0.05)
        ws_close_completed.set()

    ws.close = _slow_ws_close

    session = MagicMock()
    session.closed = False
    session.close = AsyncMock()

    adapter._ws = ws
    adapter._session = session

    async def _handshake_cancelled_mid_cleanup():
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await adapter._cleanup_ws()
            raise

    task = asyncio.ensure_future(_handshake_cancelled_mid_cleanup())
    await asyncio.sleep(0)
    task.cancel()  # cancel #1: enters the CancelledError handler
    await asyncio.wait_for(ws_close_started.wait(), timeout=2)
    # Snapshot the shielded _close_both() task now (it must already exist
    # -- ws_close_started is set from inside it) so we can wait for the
    # WHOLE unit -- both the WS close AND the session close after it -- to
    # finish, deterministically, instead of guessing from timing.
    tracked = tuple(adapter._teardown_tasks)
    task.cancel()  # cancel #2: races in while the WS close is still running
    with pytest.raises(asyncio.CancelledError):
        await task

    # _close_both() runs the WS close THEN the session close as one
    # shielded background unit. ws_close_started/ws_close_completed only
    # signal the FIRST step; they do NOT guarantee the scheduler has
    # already run the second step (the session close) by the time control
    # resumes here on every event loop implementation. Waiting on that
    # partial-progress signal and then immediately asserting on the
    # session close is exactly what made this test flaky on Linux CI
    # (observed: session.close awaited 0 times) while passing on Windows
    # by scheduling coincidence. Awaiting the tracked task itself has no
    # such gap -- see _await_tracked_teardown.
    await _await_tracked_teardown(tracked)
    assert ws_close_completed.is_set(), "the WS close must have completed"
    session.close.assert_awaited_once()
    assert adapter._ws is None
    assert adapter._session is None


@pytest.mark.asyncio
async def test_disconnect_rest_session_not_double_closed_on_cancellation():
    """disconnect() must detach self._rest_session before awaiting its
    close (#67470 review round 2, egilewski): clearing the field only
    *after* the close returned meant a cancellation racing in on
    disconnect() left the field still assigned to a session whose close was
    already running in the background -- a later disconnect() retry would
    see closed=False and call close() on it a second time."""
    adapter = _make_adapter()
    adapter._running = True

    rest_session = MagicMock()
    rest_session.closed = False
    close_started = asyncio.Event()
    close_completed = asyncio.Event()
    close_calls = 0

    async def _slow_close():
        nonlocal close_calls
        close_calls += 1
        close_started.set()
        await asyncio.sleep(0.05)
        rest_session.closed = True
        close_completed.set()

    rest_session.close = _slow_close
    adapter._rest_session = rest_session

    async def _noop():
        return

    adapter._listen_task = asyncio.ensure_future(_noop())
    adapter._watchdog_task = asyncio.ensure_future(_noop())
    await asyncio.sleep(0)

    task = asyncio.ensure_future(adapter.disconnect())
    await asyncio.wait_for(close_started.wait(), timeout=2)
    # Snapshot the shielded _full_teardown() task now (it must already
    # exist -- close_started is set from inside it, via the REST close it
    # runs directly) so we can wait for it to fully finish at the end,
    # deterministically. Deliberately NOT drained here yet -- the whole
    # point of this test is to retry disconnect() while this first close
    # may still be in flight in the background.
    tracked = tuple(adapter._teardown_tasks)
    task.cancel()  # races in while the REST session close is still running
    with pytest.raises(asyncio.CancelledError):
        await task

    # A retried disconnect() (or any other caller) must see the field
    # already cleared, not call close() on the same session a second time.
    # Safe to assert immediately, no race: _full_teardown() clears
    # self._rest_session synchronously before the close is even entered --
    # close_started already firing above proves that already happened.
    assert adapter._rest_session is None

    # Simulate a caller retrying disconnect() right away, before the
    # backgrounded close has even finished.
    adapter._running = True
    adapter._listen_task = asyncio.ensure_future(_noop())
    adapter._watchdog_task = asyncio.ensure_future(_noop())
    await asyncio.sleep(0)
    await asyncio.wait_for(adapter.disconnect(), timeout=2)

    # Deterministically wait for the FIRST close (still possibly running in
    # the background from the cancelled call above) to finish before the
    # final assert. See _await_tracked_teardown (Linux CI determinism).
    await _await_tracked_teardown(tracked)
    assert close_completed.is_set(), "the REST session close must have completed"
    assert close_calls == 1, "REST session close() must not run twice"


@pytest.mark.asyncio
async def test_disconnect_closes_rest_session_when_cancelled_during_ws_close():
    """disconnect() has several sequential stages (cancel watchdog, cancel
    listener, close WS/session, close REST session). A cancellation landing
    at an EARLIER stage must not leave a LATER stage's resources untouched
    (#67470 review round 3, egilewski): previously, cancelling disconnect()
    while _cleanup_ws() was still closing the WebSocket aborted the whole
    coroutine before the REST session close was ever attempted, leaving
    self._rest_session assigned and open. The full teardown must run as one
    protected unit so every stage still completes in the background."""
    adapter = _make_adapter()
    adapter._running = True

    ws = MagicMock()
    ws.closed = False
    ws_close_started = asyncio.Event()

    async def _slow_ws_close():
        ws_close_started.set()
        await asyncio.sleep(0.05)

    ws.close = _slow_ws_close
    adapter._ws = ws
    adapter._session = None

    rest_session = MagicMock()
    rest_session.closed = False
    rest_close_completed = asyncio.Event()

    async def _rest_close():
        await asyncio.sleep(0.02)
        rest_close_completed.set()

    rest_session.close = _rest_close
    adapter._rest_session = rest_session

    async def _noop():
        return

    adapter._listen_task = asyncio.ensure_future(_noop())
    adapter._watchdog_task = asyncio.ensure_future(_noop())
    await asyncio.sleep(0)

    task = asyncio.ensure_future(adapter.disconnect())
    await asyncio.wait_for(ws_close_started.wait(), timeout=2)
    # Snapshot the shielded task(s) now: disconnect()'s own _full_teardown()
    # task, plus _cleanup_ws()'s nested _close_both() task (both already
    # exist -- ws_close_started fires from inside the innermost one). Wait
    # for ALL of them so we know the whole nested teardown -- WS close,
    # session close, and the REST close after it -- has actually finished,
    # not just that the WS close's own event fired.
    tracked = tuple(adapter._teardown_tasks)
    task.cancel()  # races in while the WS close (an earlier stage) is running
    with pytest.raises(asyncio.CancelledError):
        await task

    # rest_close_completed alone would still be a correct (if slower to
    # fail) bound here -- if the REST close never ran, this would legitimately
    # time out rather than pass early -- but await the tracked tasks
    # directly for the same reason as the other hardened tests: no reliance
    # on scheduling order between nested shielded steps.
    await _await_tracked_teardown(tracked)
    assert rest_close_completed.is_set(), "the REST session close must have completed"
    assert adapter._rest_session is None


# ---------------------------------------------------------------------------
# Bounded-abandon teardown mechanism (#67470, Sol xhigh exhaustive mechanism
# review): an exhaustive close-site/failure-mode/state-machine enumeration
# concluded the previously-fixed `wait_for` cancellation-suppression bug
# (egilewski) was one member of a larger class. These regressions cover the
# rest of that class: `_bounded_close` must ABANDON a close after
# `_DRAIN_TIMEOUT` instead of waiting for its cancellation to actually
# finish; a close() that raises CancelledError on its own must not abort
# the surrounding cleanup sequence; the two ad-hoc ClientSession sites in
# send()/_standalone_send() (S5/S6) must be bounded and closed even under
# cancellation instead of relying on an unbounded implicit __aexit__; and a
# cold connect() superseded by disconnect() must not publish/leak sessions.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bounded_close_abandons_cancellation_suppressing_close(monkeypatch):
    """A close() that catches its own cancellation and keeps polling a stop
    signal (rather than actually finishing) must not be AWAITED to
    completion by _bounded_close.

    Pre-fix, `asyncio.wait_for(closeable.close(), timeout=_DRAIN_TIMEOUT)`
    cancels close() on timeout and then WAITS for that cancellation to
    actually finish -- so a close() that swallows the cancellation and
    keeps running left `_bounded_close` pending until close() eventually
    decides to stop on its own, which is exactly the boundedness violation
    the bounded-abandon mechanism removes (#67470 Sol xhigh mechanism
    review, gap 1)."""
    adapter = _make_adapter()
    monkeypatch.setattr(ha_adapter, "_DRAIN_TIMEOUT", 0.05, raising=False)

    stop = asyncio.Event()
    close_truly_finished = asyncio.Event()

    async def _suppresses_cancellation_until_stopped():
        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), timeout=0.05)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                continue  # rude close(): swallows cancellation, keeps going
        close_truly_finished.set()

    closeable = MagicMock()
    closeable.close = AsyncMock(side_effect=_suppresses_cancellation_until_stopped)

    # Run _bounded_close as its OWN task (mirrors the bounded-abandon
    # mechanism itself) and observe it with asyncio.wait(timeout=...)
    # rather than wrapping the bare coroutine in another wait_for.
    # asyncio.wait_for cancels the CURRENT task on timeout, not a separate
    # one, for a bare coroutine argument -- nesting it around another
    # wait_for(bare coroutine) that itself swallows cancellation forever
    # (the exact defect under test) would make the outer wait_for's own
    # cancellation land in that same swallow loop too and never escape,
    # hanging the TEST instead of failing it. asyncio.wait() on an actual
    # Task has no such trap: it only observes, never cancels.
    bounded_close_task = asyncio.ensure_future(
        adapter._bounded_close(closeable, "suppressing")
    )
    try:
        done, pending = await asyncio.wait({bounded_close_task}, timeout=1)

        assert not pending, (
            "_bounded_close must return on its own within a bounded window "
            "instead of hanging behind a cancellation-suppressing close "
            "(pre-fix: asyncio.wait_for waits for the close's actual "
            "completion after cancelling it, which a suppressing close "
            "never delivers)"
        )
        assert not close_truly_finished.is_set(), (
            "_bounded_close must abandon a cancellation-suppressing close "
            "after _DRAIN_TIMEOUT instead of waiting for it to actually "
            "finish"
        )

        # Corroborate durable retention alongside boundedness (Sol xhigh
        # mechanism re-review: the first version of this test "does not
        # prove module-level retention because it never forces GC after
        # abandonment"; the second version kept a strong local reference
        # to the task throughout, so surviving gc.collect() proved
        # nothing about the registry -- it would have survived from the
        # local alone). Find the abandoned inner close task (separate
        # from bounded_close_task, the outer _bounded_close() call, which
        # already completed above) in the module registry, take only a
        # WEAK reference, drop every strong local including the list
        # itself, force a collection, and confirm the weakref is still
        # alive. NOTE (Sol xhigh follow-up, negative-control probe): this
        # is corroborating evidence, not an isolated proof that
        # _TEARDOWN_REGISTRY specifically is what kept it alive -- a task
        # that is still actively scheduled (a live callback pending on
        # its current await) can also survive collection via asyncio's
        # own internal bookkeeping, independent of any registry. The
        # DEFINITIVE proof of the retention mechanism is the direct `in
        # _TEARDOWN_REGISTRY` membership check in
        # test_abandoned_reconnect_does_not_publish_after_disconnect
        # below; this assertion is a secondary sanity check, not the
        # sole evidence.
        abandoned = [t for t in ha_adapter._TEARDOWN_REGISTRY if not t.done()]
        assert abandoned, "the abandoned close task must be rooted in the module registry"
        abandoned_ref = weakref.ref(abandoned[0])
        del abandoned
        gc.collect()
        still_alive = abandoned_ref()
        assert still_alive is not None, (
            "a forced gc.collect() destroyed the abandoned close task after "
            "every local strong reference was dropped"
        )
        assert not still_alive.done(), (
            "a forced gc.collect() must not destroy an abandoned close "
            "that is still rooted in _TEARDOWN_REGISTRY"
        )
    finally:
        # Test hygiene: let the abandoned close actually finish instead of
        # leaving a zombie task behind -- unconditionally, even if an
        # assertion above failed (on pre-fix code the close is genuinely
        # still running at that point; without this in `finally`, the
        # loop's own teardown would gather it, and it never finishes on
        # its own since `stop` was never set, hanging the WHOLE suite
        # instead of just failing this one test).
        stop.set()
        await asyncio.wait_for(close_truly_finished.wait(), timeout=2)


@pytest.mark.asyncio
async def test_cleanup_ws_session_close_runs_when_ws_close_raises_cancellederror():
    """A close() that raises CancelledError ON ITS OWN -- not from an
    external task.cancel() -- must not abort the surrounding cleanup
    sequence: the session close must still run after it (#67470 Sol xhigh
    mechanism review, gap 3). Pre-fix, `_bounded_close` awaited
    `closeable.close()` inline inside `_close_both()`; a self-raised
    CancelledError propagated straight out of `_close_both()`, skipping
    the session close entirely."""
    adapter = _make_adapter()

    ws = MagicMock()
    ws.closed = False

    async def _raises_cancelled_from_close():
        raise asyncio.CancelledError("close() itself raises, not externally cancelled")

    ws.close = AsyncMock(side_effect=_raises_cancelled_from_close)

    session = MagicMock()
    session.closed = False
    session.close = AsyncMock()

    adapter._ws = ws
    adapter._session = session

    await asyncio.wait_for(adapter._cleanup_ws(), timeout=2)

    session.close.assert_awaited_once()
    assert adapter._ws is None
    assert adapter._session is None


@pytest.mark.asyncio
async def test_disconnect_rest_close_runs_when_ws_close_raises_cancellederror():
    """The same close-originated-CancelledError protection must hold across
    the whole disconnect() sequence, not just _cleanup_ws()'s own two
    closes: the REST session close must still run after a WS close that
    raises CancelledError on its own (#67470 Sol xhigh mechanism review,
    gap 3)."""
    adapter = _make_adapter()
    adapter._running = True

    ws = MagicMock()
    ws.closed = False

    async def _raises_cancelled_from_close():
        raise asyncio.CancelledError("close() itself raises, not externally cancelled")

    ws.close = AsyncMock(side_effect=_raises_cancelled_from_close)
    adapter._ws = ws
    adapter._session = None

    rest_session = MagicMock()
    rest_session.closed = False
    rest_session.close = AsyncMock()
    adapter._rest_session = rest_session

    async def _noop():
        return

    adapter._listen_task = asyncio.ensure_future(_noop())
    adapter._watchdog_task = asyncio.ensure_future(_noop())
    await asyncio.sleep(0)

    await asyncio.wait_for(adapter.disconnect(), timeout=2)

    rest_session.close.assert_awaited_once()
    assert adapter._rest_session is None


@pytest.mark.asyncio
async def test_send_fallback_session_closed_when_cancelled_mid_aexit(monkeypatch):
    """S5 (send()'s fallback session used when no persistent
    _rest_session exists) must still close the session when cancelled
    while the RESPONSE context's own __aexit__ is still running -- not
    just while the request itself is in flight (#67470 Sol xhigh
    mechanism review, close site S5, state-machine row "inner HTTP-
    response __aexit__ hangs before session exit"). The old `async with
    aiohttp.ClientSession() as session:` relies on the implicit
    `__aexit__` -> `await session.close()`; a cancellation landing while
    an INNER __aexit__ (the response context) is running unwinds straight
    past that outer __aexit__ too, on pre-fix code. The fix adopts the
    session explicitly and closes it from a try/finally, which runs
    regardless of where inside the try the cancellation actually lands."""
    adapter = _make_adapter()

    aexit_started = asyncio.Event()

    class _HangsOnExitPostCtx:
        async def __aenter__(self):
            resp = MagicMock()
            resp.status = 200
            return resp

        async def __aexit__(self, *exc_info):
            aexit_started.set()
            await asyncio.Event().wait()  # hang mid-__aexit__

    mock_session = MagicMock()
    mock_session.post = MagicMock(return_value=_HangsOnExitPostCtx())
    mock_session.close = AsyncMock()

    with patch("plugins.platforms.homeassistant.adapter.aiohttp") as mock_aiohttp:
        mock_aiohttp.ClientSession = MagicMock(return_value=mock_session)
        mock_aiohttp.ClientTimeout = lambda total: total

        task = asyncio.ensure_future(adapter.send("ha_events", "hi"))
        await asyncio.wait_for(aexit_started.wait(), timeout=2)
        task.cancel()  # lands inside the response context's own __aexit__
        with pytest.raises(asyncio.CancelledError):
            await task

    mock_session.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_standalone_send_session_closed_when_cancelled_mid_aexit():
    """S6 (_standalone_send()'s ad-hoc session) must still close the
    session when cancelled while the response context's own __aexit__ is
    running -- same reasoning as S5 (#67470 Sol xhigh mechanism review,
    close site S6)."""
    aexit_started = asyncio.Event()

    class _HangsOnExitPostCtx:
        async def __aenter__(self):
            resp = MagicMock()
            resp.status = 200
            return resp

        async def __aexit__(self, *exc_info):
            aexit_started.set()
            await asyncio.Event().wait()  # hang mid-__aexit__

    mock_session = MagicMock()
    mock_session.post = MagicMock(return_value=_HangsOnExitPostCtx())
    mock_session.close = AsyncMock()

    pconfig = SimpleNamespace(token="tok", extra={"url": "http://ha.local:8123"})

    with patch("plugins.platforms.homeassistant.adapter.aiohttp") as mock_aiohttp:
        mock_aiohttp.ClientSession = MagicMock(return_value=mock_session)
        mock_aiohttp.ClientTimeout = lambda total: total

        task = asyncio.ensure_future(
            ha_adapter._standalone_send(pconfig, "ha_events", "hi")
        )
        await asyncio.wait_for(aexit_started.wait(), timeout=2)
        task.cancel()  # lands inside the response context's own __aexit__
        with pytest.raises(asyncio.CancelledError):
            await task

    mock_session.close.assert_awaited_once()


def _scripted_receive_json(pause_on_first_call=None, resume_event=None):
    """Build a ``ws.receive_json`` side effect that plays the full
    auth_required -> auth_ok -> subscribe-ack handshake sequence, pausing
    on the FIRST call until *resume_event* fires if *pause_on_first_call*
    is given. Shared by the generation-race tests below so each only has
    to describe ITS pause point, not re-derive the handshake script."""
    responses = iter([
        {"type": "auth_required"},
        {"type": "auth_ok"},
        {"success": True},
    ])
    state = {"first": True}

    async def _receive_json():
        if state["first"]:
            state["first"] = False
            if pause_on_first_call is not None:
                pause_on_first_call.set()
                await resume_event.wait()
        return next(responses)

    return _receive_json


def _make_mock_ws_and_session(receive_json_side_effect):
    ws = MagicMock()
    ws.closed = False
    ws.close = AsyncMock()
    ws.send_json = AsyncMock()
    ws.receive_json = AsyncMock(side_effect=receive_json_side_effect)

    session = MagicMock()
    session.closed = False
    session.close = AsyncMock()
    session.ws_connect = AsyncMock(return_value=ws)
    return ws, session


@pytest.mark.asyncio
async def test_connect_superseded_by_disconnect_does_not_publish():
    """A disconnect() call that runs to completion while an earlier
    connect() is still cold -- paused before ``session.ws_connect()`` even
    returns, i.e. before ANYTHING has been built or published yet -- must
    prevent that connect() from later publishing sessions, starting the
    listener/watchdog, or setting _running=True (#67470 Sol xhigh
    mechanism review, gap 5).

    Drives the REAL _ws_connect() through mocked aiohttp primitives
    (rather than monkeypatching _ws_connect() itself away) so the actual
    generation-claim logic inside it runs -- a wholesale _ws_connect()
    replacement hides exactly the race this mechanism exists to close
    (Sol's REQUEST_CHANGES on the first version of this test: "patching
    _ws_connect() hides the shared-field connect/connect and
    reconnect/disconnect races"). Pauses at ``session.ws_connect()``
    itself rather than inside the auth handshake: pre-fix code published
    self._ws/self._session immediately after ws_connect() returned, BEFORE
    the handshake started, so pausing mid-handshake would have raced
    disconnect() against an ALREADY-published field on that code path --
    a coincidentally-correct result for the wrong reason, not a real test
    of "nothing published yet"."""
    adapter = _make_adapter()

    paused = asyncio.Event()
    resume = asyncio.Event()

    ws = MagicMock()
    ws.closed = False
    ws.close = AsyncMock()
    ws.send_json = AsyncMock()
    ws.receive_json = AsyncMock(side_effect=_scripted_receive_json())

    async def _paused_ws_connect(*_args, **_kwargs):
        paused.set()
        await resume.wait()
        return ws

    session = MagicMock()
    session.closed = False
    session.close = AsyncMock()
    session.ws_connect = AsyncMock(side_effect=_paused_ws_connect)

    with patch("plugins.platforms.homeassistant.adapter.aiohttp") as mock_aiohttp:
        mock_aiohttp.ClientTimeout = lambda total: total
        mock_aiohttp.ClientSession = MagicMock(return_value=session)

        connect_task = asyncio.ensure_future(adapter.connect())
        await asyncio.wait_for(paused.wait(), timeout=2)

        # disconnect() runs to completion while connect() is still cold --
        # session.ws_connect() itself hasn't even returned, so NOTHING has
        # been built yet, let alone published.
        await asyncio.wait_for(adapter.disconnect(), timeout=2)
        assert adapter._running is False
        assert adapter._session is None
        assert adapter._ws is None

        # Let the superseded connect() attempt proceed through the whole
        # handshake and reach the claim-and-publish step, which must now
        # see a stale generation.
        resume.set()
        result = await asyncio.wait_for(connect_task, timeout=2)

    assert result is False, "a superseded connect() must report failure"
    assert adapter._running is False, "superseded connect() must not flip _running back on"
    assert adapter._listen_task is None, "superseded connect() must not start a listener"
    assert adapter._watchdog_task is None, "superseded connect() must not start a watchdog"
    assert adapter._rest_session is None, "superseded connect() must not publish a REST session"
    assert adapter._ws is None, "superseded connect() must not publish the WS it opened"
    assert adapter._session is None, "superseded connect() must not publish the session it opened"

    # The WS/session built locally after resuming must be cleaned up, not
    # leaked.
    ws.close.assert_awaited_once()
    session.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_concurrent_connect_loser_closes_own_resources_not_winners():
    """Two overlapping connect() calls (#67470 Sol xhigh mechanism review,
    gap 1 -- REQUEST_CHANGES finding 1 on the first version of this fix):
    the LOSING attempt must close its OWN local ws/session, and must
    NEVER touch the fields the WINNING attempt already published.

    Pre-fix (both the original code and the first version of this
    mechanism, which published to self._ws/self._session mid-handshake
    and read them for the REST of the handshake too), two overlapping
    _ws_connect() attempts could interleave across their own handshake
    awaits and stomp on each other's shared fields: whichever finished
    LAST silently published over the other's still-live session, and the
    loser's generation-mismatch cleanup closed the SHARED fields -- which
    by then belonged to the winner -- instead of its own resources."""
    adapter = _make_adapter()

    c1_paused = asyncio.Event()
    resume_c1 = asyncio.Event()
    ws1, session1 = _make_mock_ws_and_session(_scripted_receive_json(c1_paused, resume_c1))
    ws2, session2 = _make_mock_ws_and_session(_scripted_receive_json())

    def _session_factory():
        yield session1
        yield session2
        # Anything beyond the two WS sessions above is the winner's
        # dedicated REST session (aiohttp.ClientSession(...) in
        # connect()'s success path) -- a plain throwaway is fine, this
        # test isn't asserting on it.
        while True:
            rest = MagicMock()
            rest.closed = False
            rest.close = AsyncMock()
            yield rest

    sessions = _session_factory()

    with patch("plugins.platforms.homeassistant.adapter.aiohttp") as mock_aiohttp:
        mock_aiohttp.ClientTimeout = lambda total: total
        mock_aiohttp.ClientSession = MagicMock(side_effect=lambda **_kw: next(sessions))

        c1_task = asyncio.ensure_future(adapter.connect())
        await asyncio.wait_for(c1_paused.wait(), timeout=2)

        # C2 starts and runs all the way through while C1 is still paused
        # mid-handshake -- C2 wins the race and claims the generation.
        c2_result = await asyncio.wait_for(adapter.connect(), timeout=2)
        assert c2_result is True
        assert adapter._ws is ws2
        assert adapter._session is session2

        # Let C1 finish its now-stale handshake and reach the claim step.
        resume_c1.set()
        c1_result = await asyncio.wait_for(c1_task, timeout=2)

        assert c1_result is False, "the losing connect() attempt must report failure"
        # The winner's resources must be untouched by the loser's cleanup.
        assert adapter._ws is ws2
        assert adapter._session is session2
        ws2.close.assert_not_awaited()
        session2.close.assert_not_awaited()
        # The loser must close its OWN resources, not leak them, and must
        # not have closed the winner's.
        ws1.close.assert_awaited_once()
        session1.close.assert_awaited_once()

    # Hygiene: tear down C2's now-live adapter (listener/watchdog tasks,
    # winner's WS/session) via the real disconnect() path.
    await asyncio.wait_for(adapter.disconnect(), timeout=2)


@pytest.mark.asyncio
async def test_abandoned_reconnect_does_not_publish_after_disconnect(monkeypatch):
    """A _ws_connect() attempt running inside a task that
    _cancel_task_bounded() gives up waiting for (because the handshake
    suppresses cancellation) must not publish once disconnect() has
    already moved on (#67470 Sol xhigh mechanism review, gap 5 follow-up
    -- REQUEST_CHANGES finding 2 on the first version of this fix: the
    reconnect ladder in _listen_loop calls _ws_connect() directly and had
    no generation protection at all, and the abandoned task itself wasn't
    rooted anywhere once its caller cleared its own reference).

    Exercises both halves of the fix: the abandoned task must survive
    being unrooted by its caller (checked directly against the
    module-level registry -- Sol's note that the first version's
    determinism test "does not prove module-level retention because it
    never forces GC after abandonment" applies here too), and once it
    does resume, it must back off instead of publishing
    self._session/self._ws after disconnect() already tore the adapter
    down."""
    adapter = _make_adapter()
    monkeypatch.setattr(ha_adapter, "_DRAIN_TIMEOUT", 0.05, raising=False)

    started = asyncio.Event()
    cancelled_once = asyncio.Event()
    resume_after_cancel = asyncio.Event()
    responses = iter([
        {"type": "auth_required"},
        {"type": "auth_ok"},
        {"success": True},
    ])

    async def _receive_json():
        # Plain scripted handshake -- NO cancellation swallow here. On 3.11
        # asyncio.wait_for cancels its inner future and re-raises
        # CancelledError even when the wrapped coro swallowed it, so a swallow
        # inside this wait_for-wrapped receive (adapter.py) can't survive to
        # the generation-claim step. The suppressed-cancellation wedge is
        # simulated at the BARE session.ws_connect() await below instead --
        # which is exactly the uncancellable await the mechanism exists for.
        return next(responses)

    async def _wedged_ws_connect(*_args, **_kwargs):
        # A reconnect attempt wedged on an uncancellable await (ws_connect is
        # a bare await with no local bound): swallow the cancellation
        # _cancel_task_bounded() delivers and keep running -- exactly what
        # makes it give up and abandon this task instead of waiting forever.
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled_once.set()
            await resume_after_cancel.wait()
        return ws

    ws = MagicMock()
    ws.closed = False
    ws.close = AsyncMock()
    ws.send_json = AsyncMock()
    ws.receive_json = AsyncMock(side_effect=_receive_json)

    session = MagicMock()
    session.closed = False
    session.close = AsyncMock()
    session.ws_connect = AsyncMock(side_effect=_wedged_ws_connect)

    with patch("plugins.platforms.homeassistant.adapter.aiohttp") as mock_aiohttp:
        mock_aiohttp.ClientTimeout = lambda total: total
        mock_aiohttp.ClientSession = MagicMock(return_value=session)

        reconnect_task = asyncio.ensure_future(adapter._ws_connect())
        await asyncio.wait_for(started.wait(), timeout=2)  # parked in ws_connect

        # Mirrors disconnect()'s _cancel_task_bounded(self._listen_task,
        # ...) call: cancel once, then give up after _DRAIN_TIMEOUT
        # because the wedged step suppresses it.
        await adapter._cancel_task_bounded(reconnect_task, "reconnect attempt")
        await asyncio.wait_for(cancelled_once.wait(), timeout=2)
        assert not reconnect_task.done(), (
            "the task must still be running -- abandoned, not destroyed"
        )
        assert reconnect_task in ha_adapter._TEARDOWN_REGISTRY, (
            "an abandoned task must be rooted at module scope or it can "
            "be garbage-collected before it ever reaches the generation "
            "check below"
        )

        # disconnect() proper runs next, exactly as it would have right
        # after giving up on the listen task -- moves the generation
        # forward. self._listen_task/self._watchdog_task are already
        # None here so this is a fast no-op past the generation bump.
        await asyncio.wait_for(adapter.disconnect(), timeout=2)

        # Let the abandoned reconnect finish its (suppressed-cancellation)
        # handshake and reach the claim step.
        resume_after_cancel.set()
        result = await asyncio.wait_for(reconnect_task, timeout=2)

    assert result is False, (
        "an abandoned reconnect superseded by disconnect() must not "
        "report success"
    )
    assert adapter._ws is None, "must not publish after disconnect() moved the generation"
    assert adapter._session is None
    ws.close.assert_awaited_once()
    session.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_ws_connect_rejection_cleanup_not_double_closed_on_cancellation():
    """A cancellation landing while a protocol-rejection branch's own
    cleanup is still running must not trigger a SECOND, independent close
    sequence for the same ws/session (#67470 Sol xhigh mechanism review
    follow-up on the local-handshake refactor).

    The three protocol-rejection branches (unexpected auth_required,
    auth failed, subscribe failed) call the local cleanup from INSIDE the
    handshake's try block, which shares a sibling `except
    asyncio.CancelledError:` handler. Pre-fix, a cancellation landing
    while that first cleanup call was itself awaiting its shielded close
    propagated out to the sibling handler, which called cleanup again
    with the same ws/session objects -- confirmed by Sol's own probe:
    ws.close()/session.close() each awaited twice for one failed
    connection attempt."""
    adapter = _make_adapter()

    close_started = asyncio.Event()
    close_call_count = 0

    async def _slow_ws_close():
        nonlocal close_call_count
        close_call_count += 1
        close_started.set()
        await asyncio.sleep(0.05)

    ws = MagicMock()
    ws.closed = False
    ws.close = AsyncMock(side_effect=_slow_ws_close)
    ws.send_json = AsyncMock()
    # Protocol rejection: an unexpected type on the very first receive
    # triggers the "Expected auth_required" branch immediately.
    ws.receive_json = AsyncMock(return_value={"type": "unexpected"})

    session = MagicMock()
    session.closed = False
    session.close = AsyncMock()
    session.ws_connect = AsyncMock(return_value=ws)

    with patch("plugins.platforms.homeassistant.adapter.aiohttp") as mock_aiohttp:
        mock_aiohttp.ClientTimeout = lambda total: total
        mock_aiohttp.ClientSession = MagicMock(return_value=session)

        task = asyncio.ensure_future(adapter._ws_connect())
        await asyncio.wait_for(close_started.wait(), timeout=2)
        # Snapshot the shielded close's tracked task now (it must already
        # exist -- close_started fires from inside it) so we can wait for
        # the WHOLE thing to finish afterward, deterministically (same
        # pattern as the other hardened tests in this file).
        tracked = tuple(adapter._teardown_tasks)
        task.cancel()  # lands while the rejection branch's own cleanup
        # (_close_ws_and_session, awaiting its shielded close) is running
        with pytest.raises(asyncio.CancelledError):
            await task

    await _await_tracked_teardown(tracked)
    assert close_call_count == 1, (
        "a cancellation landing mid-cleanup must not trigger a second, "
        "independent close sequence for the same ws/session"
    )
    ws.close.assert_awaited_once()
    session.close.assert_awaited_once()
