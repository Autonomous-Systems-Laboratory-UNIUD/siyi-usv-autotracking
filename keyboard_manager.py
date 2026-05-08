import sys
import termios
import select
import tty


class TerminalKeyReader:
    """
    Keyboard manager to set terminal to cbreak mode and read key pressed without blocking.
    """

    def __init__(self):
        self._fd = None  # File descriptor for standard input
        self._old_attrs = None  # To store original terminal attributes for restoration

    def __enter__(self):
        if not sys.stdin.isatty():
            return self

        self._fd = sys.stdin.fileno()
        self._old_attrs = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._fd is not None and self._old_attrs is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_attrs)

    def poll(self):
        if self._fd is None:
            return None

        ready, _, _ = select.select([sys.stdin], [], [], 0)
        if not ready:
            return None

        return sys.stdin.read(1)
