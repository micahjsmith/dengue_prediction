class Error(Exception):
    pass


class ConfigurationError(Error):
    pass


class GitError(Error):
    pass


class UnexpectedFileChangeInPullRequestError(Error):
    pass
