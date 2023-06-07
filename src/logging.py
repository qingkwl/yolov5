# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import os
import sys
import logging
import logging.config
from logging import Filter, Formatter, Logger, LogRecord, StreamHandler
from logging.handlers import RotatingFileHandler
from pathlib import Path

__all__ = [
    'set_logger',
    'get_logger'
]


ROOT_DIR = Path(__file__).parent.parent
LOGGING_NAME = "YOLOs"
MSG_FORMAT = "[%(levelname)s] %(asctime)s.%(msecs)03d [%(relativepath)s:%(lineno)d] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
FORMATTER = Formatter(fmt=MSG_FORMAT, datefmt=DATE_FORMAT)
OLD_LOG_RECORD_FACTORY = logging.getLogRecordFactory()


def log_record_factory(*args, **kwargs):
    log_record = OLD_LOG_RECORD_FACTORY(*args, **kwargs)
    pathname = log_record.pathname
    log_record.relativepath = os.path.relpath(pathname, start=ROOT_DIR)  # Enable relativepath attr in formatter
    return log_record


logging.setLogRecordFactory(log_record_factory)   # With relative path


class StdoutFilter(Filter):
    def filter(self, record: LogRecord) -> bool:
        msg = record.msg
        return len(msg) > 0    # Filter empty string


class StdoutLogger(Logger):
    def __init__(self, name, level=logging.NOTSET, stdout_level=logging.INFO):
        super(StdoutLogger, self).__init__(name=name, level=level)
        self.stdout_level = stdout_level

    def write(self, message: str):
        if message.endswith("\n"):
            message = message[:-1]  # Remove newline cause log will add extra newline
        self.log(level=self.stdout_level, msg=message)

    def flush(self):
        for handler in self.handlers:
            handler.flush()


logging.setLoggerClass(StdoutLogger)


def set_logger(logger: logging.Logger, level=logging.INFO, filename=None):
    if filename is not None:
        file_handler = RotatingFileHandler(filename=filename, maxBytes=10 * 1024 * 1024, backupCount=10)
        file_handler.setFormatter(FORMATTER)
        logger.addHandler(file_handler)
    logger.setLevel(level=level)
    logger.propagate = False
    logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
    })

    stream_handler = StreamHandler(sys.stdout)
    stream_handler.setFormatter(FORMATTER)
    logger.addHandler(stream_handler)


def get_logger(rank=None):
    rank = int(os.getenv("RANK_ID", "0")) if rank is None else rank
    name = LOGGING_NAME if rank == -1 else f"{LOGGING_NAME}_{rank}"
    logger = logging.getLogger(name)
    return logger
