"""Fail-closed windowed-mode check before a TMNF VLA episode."""

import ctypes
from concurrent.futures import Future
import sys
import threading
import time
from ctypes import wintypes
from typing import Callable


def _trackmania_windows() -> list[tuple[str, int]]:
    """Return visible TrackMania windows as ``(title, style)`` pairs."""
    if sys.platform != "win32":
        raise RuntimeError("TMNF window verification is only available on Windows")

    user32 = ctypes.windll.user32  # type: ignore[attr-defined]
    window_callback = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    user32.EnumWindows.argtypes = [window_callback, wintypes.LPARAM]
    user32.EnumWindows.restype = wintypes.BOOL
    user32.IsWindowVisible.argtypes = [wintypes.HWND]
    user32.IsWindowVisible.restype = wintypes.BOOL
    user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
    user32.GetWindowTextLengthW.restype = ctypes.c_int
    user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    user32.GetWindowTextW.restype = ctypes.c_int
    user32.GetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int]
    user32.GetWindowLongW.restype = ctypes.c_long

    matches: list[tuple[str, int]] = []

    @window_callback
    def inspect(hwnd, _):
        if not user32.IsWindowVisible(hwnd):
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        if length == 0:
            return True
        title_buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, title_buffer, length + 1)
        title = title_buffer.value
        if title.startswith("TrackMania Modded Forever"):
            matches.append((title, user32.GetWindowLongW(hwnd, -16)))
        return True

    if not user32.EnumWindows(inspect, 0):
        raise RuntimeError("could not enumerate windows")
    return matches


def assert_windowed_trackmania() -> None:
    """Require exactly one visible, titled TrackMania Modded Forever window.

    This checks the Windows style rather than trusting a command-line flag.
    It is deliberately conservative: if the window cannot be identified, an
    episode cannot start. It does not launch, resize, or focus the game.
    """
    matches = _trackmania_windows()
    if len(matches) != 1:
        raise RuntimeError(f"expected one visible TrackMania Modded Forever window; found {len(matches)}")
    title, style = matches[0]
    if style & 0x00C00000 != 0x00C00000:  # WS_CAPTION
        raise RuntimeError(f"{title!r} does not have a windowed title-bar style")


def wait_for_windowed_trackmania(
    timeout_s: float = 60.0,
    poll_s: float = 0.2,
    *,
    require_windowed: bool = True,
) -> None:
    """Wait for the game window; by default fail closed if it is not windowed.

    TMNF configured for fullscreen shows a titled window only briefly at
    startup, so unattended recording passes ``require_windowed=False``.
    """
    if timeout_s <= 0 or poll_s <= 0:
        raise ValueError("window wait durations must be positive")
    deadline = time.monotonic() + timeout_s
    while True:
        matches = _trackmania_windows()
        if matches:
            if require_windowed:
                assert_windowed_trackmania()
            elif len(matches) != 1:
                raise RuntimeError(f"expected one visible TrackMania Modded Forever window; found {len(matches)}")
            return
        if time.monotonic() >= deadline:
            raise TimeoutError("TrackMania did not create a visible window before the launch timeout")
        time.sleep(poll_s)


def _trackmania_window_for_pid(game_pid: int) -> int:
    """Return the visible TrackMania window owned by ``game_pid``."""
    if sys.platform != "win32":
        raise RuntimeError("TMNF ghost suppression is only available on Windows")
    if game_pid <= 0:
        raise ValueError("game_pid must be positive")

    user32 = ctypes.windll.user32  # type: ignore[attr-defined]
    window_callback = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    user32.EnumWindows.argtypes = [window_callback, wintypes.LPARAM]
    user32.EnumWindows.restype = wintypes.BOOL
    user32.IsWindowVisible.argtypes = [wintypes.HWND]
    user32.IsWindowVisible.restype = wintypes.BOOL
    user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
    user32.GetWindowTextLengthW.restype = ctypes.c_int
    user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    user32.GetWindowTextW.restype = ctypes.c_int
    user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    user32.GetWindowThreadProcessId.restype = wintypes.DWORD
    matches: list[int] = []

    @window_callback
    def inspect(hwnd, _):
        owner_pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner_pid))
        if owner_pid.value != game_pid or not user32.IsWindowVisible(hwnd):
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        title_buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, title_buffer, length + 1)
        if title_buffer.value.startswith("TrackMania Modded Forever"):
            matches.append(int(hwnd))
        return True

    if not user32.EnumWindows(inspect, 0):
        raise RuntimeError("could not enumerate windows")
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one visible TrackMania window for PID {game_pid}; found {len(matches)}"
        )
    return matches[0]


def _focus_window(hwnd: int) -> None:
    user32 = ctypes.windll.user32  # type: ignore[attr-defined]
    user32.SetForegroundWindow.argtypes = [wintypes.HWND]
    user32.SetForegroundWindow.restype = wintypes.BOOL
    user32.GetForegroundWindow.argtypes = []
    user32.GetForegroundWindow.restype = wintypes.HWND
    user32.SetForegroundWindow(hwnd)
    deadline = time.monotonic() + 1.0
    while user32.GetForegroundWindow() != hwnd:
        if time.monotonic() >= deadline:
            raise RuntimeError("TrackMania did not receive focus")
        time.sleep(0.01)


def bring_trackmania_to_front(game_pid: int, timeout_s: float = 3.0) -> None:
    """Restore and foreground the game window.

    A fullscreen TMNF that loses focus is minimized and stops rendering; the
    TMInterface plugin only accepts connections and delivers frames from its
    render callback, so unattended recording must keep the game in front.
    Windows only lets a background process take the foreground right after a
    key event, hence the Alt tap (the same approach linesight uses).
    """
    hwnd = _trackmania_window_for_pid(game_pid)
    user32 = ctypes.windll.user32  # type: ignore[attr-defined]
    user32.IsIconic.argtypes = [wintypes.HWND]
    user32.IsIconic.restype = wintypes.BOOL
    user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
    user32.ShowWindow.restype = wintypes.BOOL
    user32.SetForegroundWindow.argtypes = [wintypes.HWND]
    user32.SetForegroundWindow.restype = wintypes.BOOL
    user32.GetForegroundWindow.argtypes = []
    user32.GetForegroundWindow.restype = wintypes.HWND
    deadline = time.monotonic() + timeout_s
    while True:
        if user32.IsIconic(hwnd):
            user32.ShowWindow(hwnd, 9)  # SW_RESTORE
        user32.keybd_event(0x12, 0, 0, 0)  # Alt down
        user32.keybd_event(0x12, 0, 0x0002, 0)  # Alt up
        user32.SetForegroundWindow(hwnd)
        time.sleep(0.2)
        if user32.GetForegroundWindow() == hwnd:
            return
        if time.monotonic() >= deadline:
            raise RuntimeError("could not bring TrackMania to the foreground")


def _window_has_caption(hwnd: int) -> bool:
    user32 = ctypes.windll.user32  # type: ignore[attr-defined]
    user32.GetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int]
    user32.GetWindowLongW.restype = ctypes.c_long
    return user32.GetWindowLongW(hwnd, -16) & 0x00C00000 == 0x00C00000


def ensure_windowed_trackmania(game_pid: int, timeout_s: float = 60.0) -> None:
    """Switch a fullscreen TMNF to windowed mode with Alt+Enter.

    The user's TMNF always starts fullscreen. Windowed, it keeps rendering in
    the background (TMInterface's ``unfocused_fps_limit false``), so recording
    no longer needs the game to own the screen.
    """
    user32 = ctypes.windll.user32  # type: ignore[attr-defined]
    deadline = time.monotonic() + timeout_s
    while True:
        hwnd = _trackmania_window_for_pid(game_pid)
        if _window_has_caption(hwnd):
            return
        if time.monotonic() >= deadline:
            raise RuntimeError("TrackMania did not switch to windowed mode")
        bring_trackmania_to_front(game_pid)
        user32.keybd_event(0x12, 0, 0, 0)  # Alt down
        time.sleep(0.05)
        user32.keybd_event(0x0D, 0, 0, 0)  # Enter down
        time.sleep(0.05)
        user32.keybd_event(0x0D, 0, 0x0002, 0)
        user32.keybd_event(0x12, 0, 0x0002, 0)
        # The mode switch, or a still-loading game, can take a few seconds.
        time.sleep(3.0)


GAME_CLIENT_SIZE = (1280, 720)  # 16:9, matching the fullscreen recordings


def resize_trackmania_client(game_pid: int, size: tuple[int, int] = GAME_CLIENT_SIZE) -> None:
    """Give the windowed game a 16:9 drawing area; TMNF re-lays out its UI."""
    hwnd = _trackmania_window_for_pid(game_pid)
    user32 = ctypes.windll.user32  # type: ignore[attr-defined]
    user32.GetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int]
    user32.GetWindowLongW.restype = ctypes.c_long
    user32.AdjustWindowRect.argtypes = [ctypes.POINTER(wintypes.RECT), wintypes.DWORD, wintypes.BOOL]
    user32.SetWindowPos.argtypes = [
        wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint,
    ]
    left, top, right, bottom = _client_bounds(hwnd)
    if (right - left, bottom - top) == size:
        return
    rect = wintypes.RECT(0, 0, size[0], size[1])
    user32.AdjustWindowRect(ctypes.byref(rect), wintypes.DWORD(user32.GetWindowLongW(hwnd, -16) & 0xFFFFFFFF), False)
    user32.SetWindowPos(hwnd, None, 40, 40, rect.right - rect.left, rect.bottom - rect.top, 0x0004)  # SWP_NOZORDER
    time.sleep(2.0)  # let the game reset its device and re-lay out
    left, top, right, bottom = _client_bounds(hwnd)
    if (right - left, bottom - top) != size:
        raise RuntimeError(f"TrackMania client is {right - left}x{bottom - top}, expected {size[0]}x{size[1]}")


# Centre of the first profile card on "Select your Profile", as a fraction of
# the client area (measured on a 640x480 windowed client).
PROFILE_CARD_FRACTION = (0.503, 0.327)


def click_first_profile(game_pid: int) -> None:
    """Pick the (only) offline profile on TMNF's startup profile screen."""
    hwnd = _trackmania_window_for_pid(game_pid)
    user32 = ctypes.windll.user32  # type: ignore[attr-defined]
    user32.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    user32.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
    bring_trackmania_to_front(game_pid)
    client = wintypes.RECT()
    user32.GetClientRect(hwnd, ctypes.byref(client))
    point = wintypes.POINT(
        int(client.right * PROFILE_CARD_FRACTION[0]),
        int(client.bottom * PROFILE_CARD_FRACTION[1]),
    )
    user32.ClientToScreen(hwnd, ctypes.byref(point))
    user32.SetCursorPos(point.x, point.y)
    time.sleep(0.2)
    user32.mouse_event(0x0002, 0, 0, 0, 0)  # left down
    time.sleep(0.08)
    user32.mouse_event(0x0004, 0, 0, 0, 0)  # left up


def _message_boxes_for_pid(game_pid: int) -> list[int]:
    """Visible standard dialog windows (class #32770) owned by the game."""
    user32 = ctypes.windll.user32  # type: ignore[attr-defined]
    window_callback = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    user32.EnumWindows.argtypes = [window_callback, wintypes.LPARAM]
    user32.GetClassNameW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    user32.IsWindowVisible.argtypes = [wintypes.HWND]
    dialogs: list[int] = []

    @window_callback
    def inspect(hwnd, _):
        owner = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
        if owner.value == game_pid and user32.IsWindowVisible(hwnd):
            name = ctypes.create_unicode_buffer(32)
            user32.GetClassNameW(hwnd, name, 32)
            if name.value == "#32770":
                dialogs.append(int(hwnd))
        return True

    user32.EnumWindows(inspect, 0)
    return dialogs


def close_trackmania(timeout_s: float = 20.0) -> int:
    """Exit every running TmForever: ask its window to close, then terminate.

    Used between unattended laps so each lap starts from a fresh game.
    Returns how many processes were closed.
    """
    import psutil

    user32 = ctypes.windll.user32  # type: ignore[attr-defined]
    user32.PostMessageW.argtypes = [wintypes.HWND, ctypes.c_uint, wintypes.WPARAM, wintypes.LPARAM]
    processes = []
    for process in psutil.process_iter(("pid", "name")):
        if (process.info["name"] or "").startswith("TmForever"):
            processes.append(process)
            try:
                user32.PostMessageW(_trackmania_window_for_pid(process.pid), 0x0010, 0, 0)  # WM_CLOSE
            except RuntimeError:
                pass  # no window yet/anymore: fall through to terminate
    # TMNF answers WM_CLOSE with an "Are you sure you want to quit?" Yes/No
    # message box; confirm it with IDYES until the processes exit.
    deadline = time.monotonic() + timeout_s
    alive = processes
    while alive and time.monotonic() < deadline:
        for process in alive:
            for dialog in _message_boxes_for_pid(process.pid):
                user32.PostMessageW(dialog, 0x0111, 6, 0)  # WM_COMMAND, IDYES
        _, alive = psutil.wait_procs(alive, timeout=0.5)
    for process in alive:
        try:
            process.terminate()
        except psutil.NoSuchProcess:
            pass
    _, still_alive = psutil.wait_procs(alive, timeout=10)
    for process in still_alive:
        try:
            process.kill()
        except psutil.NoSuchProcess:
            pass
    return len(processes)


def _press_space(hwnd: int) -> None:
    user32 = ctypes.windll.user32  # type: ignore[attr-defined]
    user32.keybd_event(0x20, 0, 0, 0)
    time.sleep(0.08)
    user32.keybd_event(0x20, 0, 0x0002, 0)


def _client_bounds(hwnd: int) -> tuple[int, int, int, int]:
    """Screen bounds of the drawing area (no title bar/borders).

    The dialog ratios were measured on fullscreen, where window and client
    coincide; using the client keeps them valid for a windowed game.
    """
    user32 = ctypes.windll.user32  # type: ignore[attr-defined]
    user32.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    user32.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
    rect = wintypes.RECT()
    if not user32.GetClientRect(hwnd, ctypes.byref(rect)):
        raise RuntimeError("could not read TrackMania client bounds")
    origin = wintypes.POINT(0, 0)
    user32.ClientToScreen(hwnd, ctypes.byref(origin))
    return origin.x, origin.y, origin.x + rect.right, origin.y + rect.bottom


def _window_image(hwnd: int):
    """Capture the game's drawing area for dialog recognition."""
    from PIL import ImageGrab

    return ImageGrab.grab(bbox=_client_bounds(hwnd), all_screens=True).convert("RGB")


def _dialog_green_groups(image) -> list[int]:
    """Find the green title and selected-row bands in the opponent dialog."""
    width, height = image.size
    pixels = image.load()
    x0, x1 = int(width * 0.32), int(width * 0.68)
    y0, y1 = int(height * 0.18), int(height * 0.56)
    active_rows: list[int] = []
    threshold = int((x1 - x0) * 0.32)
    for y in range(y0, y1):
        green = 0
        for x in range(x0, x1):
            red, channel_green, blue = pixels[x, y]
            if (
                channel_green >= 70
                and channel_green - red >= 25
                and channel_green - blue >= 20
            ):
                green += 1
        if green >= threshold:
            active_rows.append(y)

    groups: list[list[int]] = []
    for y in active_rows:
        if not groups or y > groups[-1][-1] + 1:
            groups.append([y])
        else:
            groups[-1].append(y)
    return [group[len(group) // 2] for group in groups if len(group) >= 3]


def _opponent_dialog_visible(image) -> bool:
    """Recognize the dialog title even when no opponent row is selected."""
    height = image.height
    title_top = int(height * 0.20)
    title_bottom = int(height * 0.31)
    return any(
        title_top <= center_y <= title_bottom
        for center_y in _dialog_green_groups(image)
    )


def _window_click(hwnd: int, x_ratio: float, y_ratio: float) -> None:
    user32 = ctypes.windll.user32  # type: ignore[attr-defined]
    left, top, right, bottom = _client_bounds(hwnd)
    x = left + round((right - left) * x_ratio)
    y = top + round((bottom - top) * y_ratio)
    user32.SetCursorPos(x, y)
    user32.mouse_event(0x0002, 0, 0, 0, 0)
    time.sleep(0.06)
    user32.mouse_event(0x0004, 0, 0, 0, 0)


def _show_your_ghost_enabled(image) -> bool:
    width, height = image.size
    pixels = image.load()
    cx, cy = int(width * 0.326), int(height * 0.704)
    green = red = 0
    for y in range(max(0, cy - 8), min(height, cy + 9)):
        for x in range(max(0, cx - 8), min(width, cx + 9)):
            r, g, b = pixels[x, y]
            green += int(g > r + 20 and g > b + 15)
            red += int(r > g + 15 and r > b + 10)
    return green > red


# "None" row of the Select Opponent dialog, left of the console overlay,
# measured on a 1280x720 client: row spans x 465-797, centre y 515.
NONE_OPPONENT_ROW = (0.395, 0.715)
NONE_CLICK_ATTEMPTS = 4
# Top-left corner of the client: outside the Select Opponent dialog
# (x 0.33-0.67), so a click there can never choose an opponent.
LOAD_SCREEN_DISMISS_CLICK = (0.03, 0.05)


def _configure_opponent_dialog(game_pid: int, timeout_s: float = 45.0, none_row: tuple[float, float] = NONE_OPPONENT_ROW) -> bool:
    """Pass the load screens and choose "None" in Select Opponent."""
    hwnd = _trackmania_window_for_pid(game_pid)
    # A plain SetForegroundWindow is refused while another app is in front;
    # the Alt-tap variant works for an unattended background game.
    bring_trackmania_to_front(game_pid)
    deadline = time.monotonic() + timeout_s
    time.sleep(0.75)

    while time.monotonic() < deadline:
        image = _window_image(hwnd)
        if _opponent_dialog_visible(image):
            break
        # Dismiss "Press any key to continue" with a click in an empty corner.
        # Never press a key here: if the dialog has just appeared, Space/Enter
        # picks its highlighted row (Nadeo) and starts a race with that ghost.
        _window_click(hwnd, *LOAD_SCREEN_DISMISS_CLICK)
        time.sleep(0.35)
    else:
        raise TimeoutError("Select Opponent Ghosts dialog did not appear")

    # TMNF's "Select Opponent" is a pick-one list: Nadeo, Gold, Silver,
    # Bronze, None. Clicking a row chooses it and starts the map, so pick
    # "None". Click the row's left side: TMInterface's console overlay can
    # cover its right half. Ratios are for the 16:9 client (1280x720).
    # A click during the dialog's fade-in is ignored, so let it settle and
    # click again while it is still showing. The recorder answers the race's
    # synchronous steps on its own thread, so checking here cannot deadlock:
    # a paused game simply shows the countdown instead of the dialog.
    for _ in range(NONE_CLICK_ATTEMPTS):
        time.sleep(0.8)
        bring_trackmania_to_front(game_pid)
        _window_click(hwnd, none_row[0], none_row[1])
        time.sleep(2.0)
        if not _opponent_dialog_visible(_window_image(hwnd)):
            return True
    raise RuntimeError("Select Opponent dialog ignored the None click")


class OpponentDialogController:
    """Configure a solo map with no medal/downloaded/PB ghosts."""

    def __init__(
        self,
        configure: Callable[[int], bool] = _configure_opponent_dialog,
        none_row_y: float | None = None,
    ):
        # Maps without a Nadeo ghost list one row fewer, so "None" sits higher
        # (A01: 0.715 with five rows; A03: 0.62 with four).
        if none_row_y is not None:
            configure = lambda pid: _configure_opponent_dialog(pid, none_row=(NONE_OPPONENT_ROW[0], none_row_y))
        self.configure = configure

    def schedule_for_process(self, game_pid: int) -> Future[bool]:
        future: Future[bool] = Future()

        def run() -> None:
            try:
                future.set_result(self.configure(game_pid))
            except BaseException as exc:
                future.set_exception(exc)

        threading.Thread(target=run, name="tmnf-opponent-dialog", daemon=True).start()
        return future
