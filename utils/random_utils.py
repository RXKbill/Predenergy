import os
import random
from typing import Optional, NoReturn

import numpy as np
import paddle


def fix_random_seed(seed: Optional[int] = 2021) -> NoReturn:
    if seed is None:
        return

    random.seed(seed)
    paddle.seed(seed)
    np.random.seed(seed)


def fix_all_random_seed(seed: Optional[int] = 2021) -> NoReturn:
    if seed is None:
        return

    paddle.seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    os.environ["PYTHONHASHSEED"] = str(1)
