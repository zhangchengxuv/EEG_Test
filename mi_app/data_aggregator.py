import numpy as np

class BlockAggregator:
    def __init__(self, block_size):
        """
        block_size: 目标 block 大小（如 64）
        """
        self.block_size = block_size
        self.buffer = np.zeros(0, dtype=float)

    def add(self, data_16):
        """
        data_16: shape (16,)
        返回: list of blocks，每个 block shape (block_size,)
        """
        self.buffer = np.concatenate([self.buffer, data_16])

        blocks = []
        while len(self.buffer) >= self.block_size:
            block = self.buffer[:self.block_size]
            blocks.append(block)
            self.buffer = self.buffer[self.block_size:]

        return blocks

    def reset(self):
        self.buffer = np.zeros(0, dtype=float)
