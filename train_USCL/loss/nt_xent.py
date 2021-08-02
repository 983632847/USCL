import torch
import numpy as np


class NTXentLoss(torch.nn.Module):

    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum") # sum all 2N terms of loss instead of getting mean val

    def _get_similarity_function(self, use_cosine_similarity):
        ''' Cosine similarity or dot similarity for computing loss '''
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size) # I(2Nx2N), identity matrix
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size) # lower diagonal matrix, N non-zero elements
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size) # upper diagonal matrix, N non-zero elements
        mask = torch.from_numpy((diag + l1 + l2)) # [2N, 2N], with 4N elements are non-zero
        mask = (1 - mask).type(torch.bool) # [2N, 2N], with 4(N^2 - N) elements are "True"
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2) # extend the dimensions before calculating similarity 
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C), N input samples
        # y shape: (1, 2N, C), 2N output representations
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0)) # extend the dimensions before calculating similarity 
        return v

    def forward(self, zis, zjs):
        if self.batch_size != zis.shape[0]:
            self.batch_size = zis.shape[0] # the last batch may not have the same batch size
    
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        representations = torch.cat([zjs, zis], dim=0) # [N, C] => [2N, C]

        similarity_matrix = self.similarity_function(representations, representations) # [2N, 2N]

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size) # upper diagonal, N x [left, right] positive sample pairs
        r_pos = torch.diag(similarity_matrix, -self.batch_size) # lower diagonal, N x [right, left] positive sample pairs
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1) # similarity of positive pairs, [2N, 1]

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1) # [2N, 2N]

        logits = torch.cat((positives, negatives), dim=1) # [2N, 2N+1], the 2N+1 elements of one column are used for one loss term
        logits /= self.temperature

        # labels are all 0, meaning the first value of each vector is the nominator term of CELoss
        # each denominator contains 2N+1-2 = 2N-1 terms, corresponding to all similarities between the sample and other samples.
        labels = torch.zeros(2 * self.batch_size).to(self.device).long() 
        loss = self.criterion(logits, labels)

        return loss / (2 * self.batch_size) # Don't know why it is divided by 2N, the CELoss can set directly to reduction='mean'
















