import numpy as np

from logger import Logger

logger = Logger()
logger.extend(
    {
        "loss/train": np.random.randn(15),
        "loss/eval": np.random.randn(10),
    },
)
logger.step()

logger.extend(
    {
        "loss/train": np.random.randn(15),
        "loss/eval": np.random.randn(10),
    },
)
logger.step()

logger.save("logger.npz")
