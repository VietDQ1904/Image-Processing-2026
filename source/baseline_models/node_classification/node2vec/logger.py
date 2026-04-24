import torch

class Logger(object):
    def __init__(self, info=None, log_path=None):
        self.info = info
        self.log_path = log_path

    def _log(self, text):
        print(text)
        if self.log_path is not None:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(text + "\n")

