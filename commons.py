
import os
import sys
import logging
import traceback


def setup_logging(output_folder: str, stdout: str = "debug", info_filename: str = "info.log",
                  debug_filename: str = "debug.log"):
    """After calling this function, you can easily log messages from anywhere
    in your code without passing any object to your functions.
    Just calling "logging.info(msg)" prints "msg" in stdout and saves it in
    the "info.log" and "debug.log" files.
    Similarly, "logging.debug(msg)" saves "msg" in the "debug.log" file.

    Parameters
    ----------
    output_folder : str, the folder where to save the logging files.
    stdout : str, can be "debug" or "info".
        If stdout == "debug", print in stdout any time logging.debug(msg)
        (or logging.info(msg)) is called.
        If stdout == "info", print in std out only when logging.info(msg) is called.
    info_filename : str, name of the file with the logs printed when calling
        logging.info(msg).
    debug_filename : str, name of the file with the logs printed when calling
        logging.debug(msg) or logging.info(msg).

    """
    if os.path.exists(output_folder):
        raise FileExistsError(f"{output_folder} already exists!")
    os.makedirs(output_folder)
    # logging.Logger.manager.loggerDict.keys() to check which loggers are in use
    logging.getLogger('matplotlib.font_manager').disabled = True
    logging.getLogger('shapely').disabled = True
    logging.getLogger('shapely.geometry').disabled = True
    base_formatter = logging.Formatter('%(asctime)s   %(message)s', "%Y-%m-%d %H:%M:%S")
    logger = logging.getLogger('')
    logger.setLevel(logging.DEBUG)
    logging.getLogger('PIL').setLevel(logging.INFO)  # turn off logging tag for some images

    if info_filename is not None:
        info_file_handler = logging.FileHandler(f'{output_folder}/{info_filename}')
        info_file_handler.setLevel(logging.INFO)
        info_file_handler.setFormatter(base_formatter)
        logger.addHandler(info_file_handler)

    if debug_filename is not None:
        debug_file_handler = logging.FileHandler(f'{output_folder}/{debug_filename}')
        debug_file_handler.setLevel(logging.DEBUG)
        debug_file_handler.setFormatter(base_formatter)
        logger.addHandler(debug_file_handler)

    if stdout is not None:
        console_handler = logging.StreamHandler()
        if stdout == "debug":
            console_handler.setLevel(logging.DEBUG)
        if stdout == "info":
            console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(base_formatter)
        logger.addHandler(console_handler)

    # Save exceptions in log files
    def exception_handler(type_, value, tb):
        logger.info("\n" + "".join(traceback.format_exception(type, value, tb)))

    sys.excepthook = exception_handler

