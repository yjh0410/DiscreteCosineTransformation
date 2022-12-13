import numpy as np
import torch


class DCTransform(object):
    """A numpy deployment of 2D Discrete Cosine Transform (DCT) to process a batch of 2D images"""
    def __init__(self, vmax, hmax):
        assert vmax == hmax, "only support the square image, where the vmax is equal to hmax."

        self.vmax = vmax
        self.hmax = hmax
        self.seq_len = hmax

        # generate transform matrix
        self.transform_matrix, self.const = self.generate_transform_matrix()
        self.transform_matrix_T = self.transform_matrix.T


    def generate_transform_matrix(self):
        """ generate transform matrix with [1, H, W] shape, where the '1' is
        the batch dimension."""
        # generate transform matrix
        seq_len = self.seq_len
        sequence_1 = np.arange(seq_len).reshape(seq_len, 1) # [N, 1]
        sequence_2 = ((2. * sequence_1 + 1) * np.pi).reshape(1, seq_len) / (2. * seq_len)  # [1, N]

        # [N, 1] x [1, N] = [N, N]
        transform_matrix = np.cos(np.matmul(sequence_1, sequence_2))

        # The results of the above matrix operation and the for loop below are identical, 
        # but the above matrix operation is faster.
        # transform_matrix = np.zeros([seq_len, seq_len], dtype=np.float32)
        # for i in range(seq_len):
        #     for j in range(seq_len):
        #         transform_matrix[i, j] = np.cos(((2.*j + 1) * np.pi) * i / (2.*seq_len))

        # fcstor matrix
        const = np.zeros([self.seq_len, self.seq_len], dtype=np.float32)
        const[0, 0] = 1. / seq_len
        const[0, 1:] = np.sqrt(2.) / seq_len
        const[1:, 0] = np.sqrt(2.) / seq_len
        const[1:, 1:] = 2. / seq_len

        return transform_matrix, const


    def dct(self, input):
        """
        Input:
            input: (ndarray) a batch of 2D gray images with [B, H, W] shape.
        """
        assert len(input.shape) == 3, \
            "the input should be a batch of 2D gray image with [B, H, W] shape, not [H, W] shape."

        B = input.shape[0]
        # TODO
        # [N, N] -> [1, N, N] -> [B, N, N]
        trans_m = np.repeat(self.transform_matrix[None], B, axis=0)
        trans_m_T = np.repeat(self.transform_matrix_T[None], B, axis=0)
        const_m = np.repeat(self.const[None], B, axis=0)

        feature = const_m * np.matmul(np.matmul(trans_m, input), trans_m_T)

        return feature


    def idct(self, feature):
        """
        Input:
            input: (ndarray) a batch of 2D dct images with [B, H, W] shape.
        """
        assert len(feature.shape) == 3, \
            "the input should be a batch of 2D dct image with [B, H, W] shape, not [H, W] shape."

        B = feature.shape[0]
        # TODO
        # [N, N] -> [1, N, N] -> [B, N, N]
        trans_m = np.repeat(self.transform_matrix[None], B, axis=0)
        trans_m_T = np.repeat(self.transform_matrix_T[None], B, axis=0)
        const_m = np.repeat(self.const[None], B, axis=0)

        image = const_m * np.matmul(np.matmul(trans_m_T, feature), trans_m)

        return image


class DCTransformTroch(object):
    """A pytorch deployment of 2D Discrete Cosine Transform (DCT) to process a batch of 2D images"""
    def __init__(self, vmax, hmax, device):
        assert vmax == hmax, "only support the square image, where the vmax is equal to hmax."

        self.vmax = vmax
        self.hmax = hmax
        self.seq_len = hmax
        self.device = device

        # generate transform matrix
        self.transform_matrix, self.const = self.generate_transform_matrix()
        self.transform_matrix_T = self.transform_matrix.T


    def generate_transform_matrix(self):
        """ generate transform matrix with [1, H, W] shape, where the '1' is
        the batch dimension."""
        # generate transform matrix
        seq_len = self.seq_len
        sequence_1 = torch.arange(seq_len, dtype=torch.float32).reshape(seq_len, 1) # [N, 1]
        sequence_2 = ((2. * sequence_1 + 1) * np.pi).view(1, seq_len) / (2. * seq_len)  # [1, N]

        # [N, 1] x [1, N] = [N, N]
        transform_matrix = torch.cos(sequence_1 @ sequence_2)

        # fcstor matrix
        const = torch.zeros([self.seq_len, self.seq_len], dtype=torch.float32)
        const[0, 0] = torch.tensor(1. / seq_len, dtype=torch.float32)
        const[0, 1:] = torch.tensor(np.sqrt(2.) / seq_len, dtype=torch.float32)
        const[1:, 0] = torch.tensor(np.sqrt(2.) / seq_len, dtype=torch.float32)
        const[1:, 1:] = torch.tensor(2. / seq_len, dtype=torch.float32)

        # to device
        transform_matrix = transform_matrix.to(self.device)
        const = const.to(self.device)

        return transform_matrix, const


    def dct(self, input):
        """
        Input:
            input: (ndarray) a batch of 2D gray images with [B, H, W] shape.
        """
        assert len(input.shape) == 3, \
            "the input should be a batch of 2D gray image with [B, H, W] shape, not [H, W] shape."

        B = input.shape[0]
        # TODO
        # [N, N] -> [1, N, N] -> [B, N, N]
        trans_m = self.transform_matrix[None].repeat(B, 1, 1)
        trans_m_T = self.transform_matrix_T[None].repeat(B, 1, 1)
        const_m = self.const[None].repeat(B, 1, 1)

        feature = const_m * ((trans_m @ input) @ trans_m_T)

        return feature


    def idct(self, feature):
        """
        Input:
            input: (ndarray) a batch of 2D dct images with [B, H, W] shape.
        """
        assert len(feature.shape) == 3, \
            "the input should be a batch of 2D dct image with [B, H, W] shape, not [H, W] shape."

        B = feature.shape[0]
        # TODO
        # [N, N] -> [1, N, N] -> [B, N, N]
        trans_m = self.transform_matrix[None].repeat(B, 1, 1)
        trans_m_T = self.transform_matrix_T[None].repeat(B, 1, 1)
        const_m = self.const[None].repeat(B, 1, 1)

        image = const_m * ((trans_m_T @ feature) @ trans_m)

        return image


if __name__ == '__main__':
    dct = DCTransform(hmax=80, vmax=80)
    dct_torch = DCTransformTroch(hmax=80, vmax=80, device='cpu')