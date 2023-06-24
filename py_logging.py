import os
import sys
import logging
import datetime
from logging import FileHandler

base_path = os.path.dirname(os.path.abspath(__file__))

class Formatter(logging.Formatter):
    """override logging.Formatter to use an aware datetime object"""

    def converter(self, timestamp):
        dt = datetime.datetime.fromtimestamp(timestamp)
        # tzinfo = pytz.timezone('US/Pacific')
        # return tzinfo.localize(dt)
        # return dt.replace(tzinfo=pytz.UTC)
        return dt.astimezone()

    def formatTime(self, record, datefmt=None):
        dt = self.converter(record.created)
        if datefmt:
            s = dt.strftime(datefmt)
        else:
            try:
                s = dt.isoformat(timespec='milliseconds')
            except TypeError:
                s = dt.isoformat()
        return s


logFormatter = Formatter("[%(asctime)s] :%(levelname)s: [%(filename)s:%(lineno)d], %(message)s",
                         datefmt='%Y-%m-%d %H:%M:%S.%f %Z') # %Y-%m-%d %H:%M:%S.%f %Z %z -- to add offset

def get_console_handler():
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logFormatter)
    return console_handler


def get_file_handler(LOG_FILE):
    # file_handler = TimedRotatingFileHandler(LOG_FILE) #, when='midnight')
    file_handler = FileHandler(LOG_FILE)
    file_handler.setFormatter(logFormatter)
    return file_handler


def get_logger(module_name, addl_file_name, save_to_file = True):
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.DEBUG)  # better to have too much log than not enough
    logger.addHandler(get_console_handler())

    if addl_file_name is not None:
        univ_log_file = os.path.join(base_path, "results", addl_file_name)
        pathExist = os.path.exists(os.path.split(univ_log_file)[0])
        if not pathExist:
            os.makedirs(os.path.split(univ_log_file)[0])
        LOG_FILE = univ_log_file

        if save_to_file:
            logger.addHandler(get_file_handler(LOG_FILE))
    # with this pattern, it's rarely necessary to propagate the error up to parent
    logger.propagate = False
    return logger


if __name__ == "__main__":
    addl_file_loc = "/Users/kgarg/Documents/dvt_automation/enl_stuff/results/1234/dddd/1233333.txt"
    my_logger = get_logger(__name__, addl_file_loc, save_to_file=True)
    my_logger.debug("a debug message")

