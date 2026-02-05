class CodeIndexError(Exception):
    """Base class for code index errors."""


class EmbeddingError(CodeIndexError):
    def __init__(self, message: str, status_code: int | None = None, detail: str | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.detail = detail


class GitError(CodeIndexError):
    pass


class IndexingError(CodeIndexError):
    pass
