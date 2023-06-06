import os
import logging
from datetime import datetime


class DefaultLogger:
    def info(self, msg: str):
        print(msg)


def setup_evaluation_logger(args, prefix: str = 'eval'):
    logger = logging.getLogger('让我毕业球球了.txt')
    logger.setLevel(logging.INFO)

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    ckpt_name = os.path.basename(os.path.dirname(args.checkpoint_path))

    filename = f'{prefix}-{ckpt_name}.log'

    filename = f'{timestamp}-{filename}'

    file_handler = logging.FileHandler(filename=f'./logs/{filename}',
                                       mode='a',
                                       encoding='utf-8')
    file_formatter = logging.Formatter("%(message)s")
    file_handler.setFormatter(file_formatter)

    logger.addHandler(file_handler)

    return logger


def setup_logger(args):
    logger = logging.getLogger('让我毕业球球了')
    logger.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_formatter = logging.Formatter("%(message)s")

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    filename = f'{args.dataset}-{args.seed}.log'

    if hasattr(args, 'log_prefix') and args.log_prefix != '':
        filename = f'{args.log_prefix}-{filename}'

    filename = f'{timestamp}-{filename}'

    file_handler = logging.FileHandler(filename=f'./logs/{filename}',
                                       mode='a',
                                       encoding='utf-8')
    file_formatter = logging.Formatter("[%(levelname)s %(asctime)s]: %(message)s")

    stream_handler.setFormatter(stream_formatter)
    file_handler.setFormatter(file_formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger
