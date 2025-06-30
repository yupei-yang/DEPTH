import logging
import logging.config
import os

# 指定日志文件的路径
LOG_FILE_PATH = os.path.join("logs", "PRE-PROCESS-DATASET-4-REFINEMENT-test.log")

# 检查 logs 文件夹是否存在，不存在则创建
os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)

DEFAULT_LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "general": {
            "()": "logging.Formatter",
            "fmt": "%(asctime)s %(levelname)-8s [%(module)s:%(lineno)d] %(message)s",
            "style": "%"
        },
    },
    "handlers": {
        "console": {
            "formatter": "general",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout"
        },
        "file": {
            "formatter": "general",
            "class": "logging.FileHandler",
            "filename": LOG_FILE_PATH,  # 设置日志文件路径
            "mode": "a",  # 追加模式
            "encoding": "utf-8"
        }
    },
    "loggers": {
        "util": {
            "handlers": ["console", "file"],  # 同时输出到控制台和文件
            "level": logging.DEBUG,
            "propagate": False
        }
    }
}


def setup_logging_config(dict_cfg=None):
    if dict_cfg is not None:
        logging.config.dictConfig(dict_cfg)
    else:
        logging.config.dictConfig(DEFAULT_LOGGING_CONFIG)


def get_logger(name: str):
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        return logger
    setup_logging_config()
    return logging.getLogger(name)
