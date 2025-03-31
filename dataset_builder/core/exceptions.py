class PipelineError(Exception):
    """Base class for all pipeline-related exceptions."""

    pass


class FailedOperation(PipelineError):
    """External or recoverable failure (e.g., network, disk)."""

    def __init__(self, message: str):
        super().__init__(message)


class ConfigError(PipelineError):
    """Bad or missing configuration."""

    def __init__(self, key: str):
        super().__init__(f"Missing or invalid config key: {key}")


class AnalysisError(PipelineError):
    """Failure during dataset analysis or statistics."""

    def __init__(self, reason: str):
        super().__init__(f"Dataset analysis failed: {reason}")
