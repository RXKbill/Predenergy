#!/usr/bin/env python
# -*- coding:utf-8 _*-
import logging
import os
import sys
import paddle
from typing import Optional, List


def is_local_rank_0():
    """Check if current process is local rank 0"""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    return local_rank == 0


def get_logger(name, level="INFO", handlers=None, update=False):
    """Get logger with specified name and level"""
    logger = logging.getLogger(name)
    if not update and logger.handlers:
        return logger
    
    logger.setLevel(getattr(logging, level.upper()))
    
    if handlers is None:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        handlers = [handler]
    
    for handler in handlers:
        logger.addHandler(handler)
    
    return logger


def log_in_local_rank_0(*msg, type='info', used_logger=None):
    """Log message only in local rank 0 process"""
    if not is_local_rank_0():
        return
    
    if used_logger is None:
        used_logger = get_logger("Predenergy")
    
    log_func = getattr(used_logger, type.lower(), used_logger.info)
    log_func(" ".join(str(m) for m in msg)) 