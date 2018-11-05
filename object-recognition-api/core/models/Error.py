class ErrorResponse:
    def __init__(self, error_code, message, errors=None):
        self.code = error_code
        self.message = message
        if errors is None:
            errors = []
        self.errors = errors


class Error400(ErrorResponse):
    def __init__(self):
        code = 400
        message = 'The request is malformed; ex. the body does not parse, some validation fails, semantically incorrect.'
        errors = []
        super(Error400, self).__init__(code, message, errors)


class Error404(ErrorResponse):
    def __init__(self):
        code = 404
        message = 'Resource not found.'
        errors = []
        super(Error404, self).__init__(code, message, errors)


class Error500(ErrorResponse):
    def __init__(self):
        code = 500
        message = 'Unexpected error.'
        errors = []
        super(Error500, self).__init__(code, message, errors)
