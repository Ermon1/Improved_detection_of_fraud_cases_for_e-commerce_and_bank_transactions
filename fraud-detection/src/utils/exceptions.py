import sys

def error_message_details(error: Exception) -> str:
    """
    Extracts the file name, line number, and message from the original exception.
    Returns a structured, readable string.
    """
    _, _, exc_tb = sys.exc_info()
    if exc_tb:
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_no = exc_tb.tb_lineno
    else:
        file_name, line_no = "Unknown", "Unknown"

    return "Error occurred in python script [{}] at line [{}], error message: [{}]".format(
        file_name, line_no, str(error)
    )

class CustomException(Exception):
    """
    Custom exception that wraps any standard exception and formats a detailed message.
    Does not raise a full Python traceback unless explicitly desired.
    """
    def __init__(self, error: Exception):

        self.error_message = error_message_details(error)

    def __str__(self):
        return self.error_message

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.error_message)})"

try:
    5 / 0
except Exception as e:
    raise  CustomException(e)

    # Option 2: Log structured error
    # import logging
    # logging.error(CustomException(e))
