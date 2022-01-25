import torch
import torch.nn as nn


class ComplexResGate(nn.Module):
    def __init__(self, embedding_size):
        super(ComplexResGate, self).__init__()
        self.fc1 = nn.Linear(2 * embedding_size, 2 * embedding_size)
        self.fc2 = nn.Linear(2 * embedding_size, embedding_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, m, m_tild):
        m_concat = torch.cat((m, m_tild), dim=2)
        x = self.fc1(m_concat)
        z = self.sigmoid(x)
        y = self.fc2(z * m_concat)

        return y
