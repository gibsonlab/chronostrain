import torch
from typing import List


class BlockConstantMatrix(object):
    def __init__(self, constants: torch.Tensor, rows: List[int], cols: List[int]):
        """
        :param constants: The matrix (2-d tensor) of constant values.
        :param rows: A list of row block sizes, whose size matches the # of rows of "constants".
        :param cols: A list of col block sizes, whose size matches the # of cols of "constants".
        """
        self.constants = constants
        self.rows = rows
        self.cols = cols

    def mm_blocks(self, b):
        if not isinstance(b, BlockConstantMatrix):
            raise Exception("Can only block-multiply with another BlockConstantMatrix.")
        if self.cols != b.rows:
            raise Exception("Columns of A must match rows of B.")
        return BlockConstantMatrix(
            self.constants.mm(
                torch.diag(torch.tensor(
                    self.cols
                )).mm(b.constants)
            ),
            self.rows,
            b.cols
        )

    def __repr__(self):
        return "{}[values:\n{}\nrows:{}\ncols:{}]".format(
            self.__class__.__name__,
            self.constants,
            self.rows,
            self.cols
        )

    def __str__(self):
        return self.__repr__()

    @staticmethod
    def from_torch(x: torch.Tensor):
        return BlockConstantMatrix(constants=x,
                                   rows=[1 for _ in x.size()[0]],
                                   cols=[1 for _ in x.size()[1]])

    def __getitem__(self, item):
        if not isinstance(item, tuple):
            raise Exception("Length-2 tuple expected for __getitem__, got type {}".format(type(item)))
        elif len(item) != 2:
            raise Exception("Length-2 tuple expected for __getitem__, got len {}".format(len(item)))

    def to_torch(self):
        return torch.cat([
            torch.cat([
                self.constants[i, j] * torch.ones(size=(self.rows[i], self.cols[j]))
                for j in range(len(self.cols))
            ], dim=1)
            for i in range(len(self.rows))
        ], dim=0)


# b = BlockConstantMatrix(constants=torch.tensor([[0, 1], [2, 3]]), rows=[1, 2], cols=[2, 3])
# c = BlockConstantMatrix(constants=torch.tensor([[0, 1], [2, 3]]), rows=[2, 3], cols=[4, 5])
#
#
# print(b.to_torch())
# print(c.to_torch())
# print(b.mm_blocks(c).to_torch())
# print(b.to_torch().mm(c.to_torch()))
