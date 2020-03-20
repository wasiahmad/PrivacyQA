import torch
import math
import torch.nn as nn
from torch.nn import Parameter


class MatrixAttention(nn.Module):
    """
    This ``Module`` takes two matrices as input and returns a matrix of attentions.
    We compute the similarity between each row in each matrix and return unnormalized similarity
    scores.
    Input:
        - matrix_1: ``(batch_size, num_rows_1, embedding_dim)``
        - matrix_2: ``(batch_size, num_rows_2, embedding_dim)``
    Output:
        - ``(batch_size, num_rows_1, num_rows_2)``
    """

    def __init__(self, dim):
        super(MatrixAttention, self).__init__()
        self._weight_vector = Parameter(torch.Tensor(dim * 3))
        self._bias = Parameter(torch.Tensor(1))
        self.reset_parameters()

    def reset_parameters(self):
        std = math.sqrt(6 / (self._weight_vector.size(0) + 1))
        self._weight_vector.data.uniform_(-std, std)
        self._bias.data.fill_(0)

    def forward(self, matrix_1, matrix_2):
        matrix_1 = matrix_1.unsqueeze(2)
        matrix_2 = matrix_2.unsqueeze(1)
        assert matrix_1.size(-1) == matrix_2.size(-1)
        dim = matrix_1.size(-1)

        to_sum = []
        to_sum.append(torch.matmul(matrix_1, self._weight_vector[:dim]))
        to_sum.append(torch.matmul(matrix_2, self._weight_vector[dim:dim * 2]))
        intermediate = matrix_1.squeeze(2) * self._weight_vector[dim * 2:]
        to_sum.append(torch.matmul(intermediate, matrix_2.squeeze(1).transpose(-1, -2)))

        result = to_sum[0]
        for result_piece in to_sum[1:]:
            result = result + result_piece
        return result + self._bias
