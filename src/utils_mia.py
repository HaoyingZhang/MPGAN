import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

class MIAModel(nn.Module):
    def __init__(self, T, D):
        super().__init__()
        self.lstm = nn.LSTM(D, 32, batch_first=True)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return torch.sigmoid(self.fc(h[-1]))

def run_mia_attack(member_data, non_member_data, generated_data):
    """
    Simple MIA: train a binary classifier to separate members vs non-members.
    """
    X_mem = torch.stack([member_data[i][0].unsqueeze(-1) for i in range(len(member_data))]).float()
    X_non = torch.stack([non_member_data[i][0].unsqueeze(-1) for i in range(len(non_member_data))]).float()
    y_mem = torch.ones(len(X_mem))
    y_non = torch.zeros(len(X_non))

    X_all = torch.cat([X_mem, X_non])
    y_all = torch.cat([y_mem, y_non])

    dataset = TensorDataset(X_all, y_all)
    train_set, test_set = random_split(dataset, [int(0.7 * len(dataset)), int(0.3 * len(dataset))])
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=32)

    model = MIAModel(T=X_all.shape[1], D=X_all.shape[2])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCELoss()

    # Train
    model.train()
    for epoch in range(10):
        for xb, yb in train_loader:
            pred = model(xb).squeeze()
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluate
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            pred = model(xb).squeeze() > 0.5
            correct += (pred.int() == yb.int()).sum().item()
            total += len(yb)

    return correct / total
