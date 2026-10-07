"""Execution errors shared by simulator, skills and display adapters."""

class M2Failure(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class ViewerClosed(RuntimeError):
    """Raised when the user closes the live viewer before an episode completes."""
