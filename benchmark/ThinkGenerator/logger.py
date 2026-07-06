from pathlib import Path
from functools import total_ordering
import time


# 给定str，写入指定的log文件中
class Logger:
    def __init__(self, log_file):
        self.log_file = log_file

    # 修改文件
    def set_log_file(self, log_file):
        self.log_file = log_file

    def log(self, str):
        if not Path(self.log_file).parent.exists():
            Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_file, "a") as f:
            f.write(str)
