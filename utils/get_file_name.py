import os
import socket
import time


def get_unique_file_suffix():
    # Get Host Name
    hostname = socket.gethostname()

    # Get current timestamp (seconds since Unix era)
    timestamp = int(time.time())

    # Obtain the PID (process identifier) of the process
    pid = os.getpid()

    # Build file name
    log_filename = f".{timestamp}.{hostname}.{pid}.csv"
    return log_filename
