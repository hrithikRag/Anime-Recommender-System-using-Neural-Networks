import sys

class CustomException(Exception):
    def __init__(self, message, error_detail: Exception):
        super().__init__(message)
        _, _, exc_tb = sys.exc_info()
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
        self.error_message = f"{message} | Error in {file_name}, line {line_number}: {str(error_detail)}"

    def __str__(self):
        return self.error_message
