import numpy as np

from logger import Logger

logger = Logger()
for i in range(100):
    logger.extend(
        {
            "loss/train": (1 + np.random.randn(50)) / 1e5 * np.exp(-i / 100),
            "loss/eval": (1 + np.random.randn(10)) / 1e4,
        },
    )
    logger.step()

logger.save("logger.npz")
