class IndslUserWarning(UserWarning):
    """Warning that will be shown to the user."""

    def __init__(self, message):
        self.message = message
