import torch
from torch import nn
from torch.utils.data import DataLoader
from dataloader import tensor_gen

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim=16, hidden_dim=256, output_dim=4):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.action = nn.Linear(hidden_dim // 2, output_dim)
        self.pot    = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x):
        features = self.backbone(x)
        action_logits = self.action(features)
        value_pred    = self.pot(features)
        return action_logits, value_pred.squeeze(-1)


# ==================== TRAINING & TEST LOOPS ====================
def train_loop(dataloader, model, loss_fn_1, loss_fn_2, optimizer, value_weight=0.3):
    model.train()
    for batch_idx, (X, y_action, y_value) in enumerate(dataloader):
        optimizer.zero_grad()

        action_logits, value_pred = model(X)          # ONE forward pass

        loss_1 = loss_fn_1(action_logits, y_action)
        loss_2 = loss_fn_2(value_pred, y_value)
        loss = loss_1 + value_weight * loss_2

        loss.backward()
        optimizer.step()

        if batch_idx % 50 == 0:   # print every 50 batches
            print(f"Batch {batch_idx:4d} | Loss: {loss.item():.4f}")

def test_loop(dataloader, model, loss_fn_1, loss_fn_2, value_weight=0.3):
    model.eval()
    total_loss = 0
    correct = 0
    size = len(dataloader.dataset)

    with torch.no_grad():
        for X, y_action, y_value in dataloader:
            action_logits, value_pred = model(X)
            loss_1 = loss_fn_1(action_logits, y_action)
            loss_2 = loss_fn_2(value_pred, y_value)
            total_loss += (loss_1 + value_weight * loss_2).item()

            pred_class = action_logits.argmax(dim=1)
            correct += (pred_class == y_action).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / size
    print(f"Test  → Accuracy: {accuracy:>6.1f}% | Avg loss: {avg_loss:>8.4f}\n")


# ==================== RUN TRAINING ====================
model = NeuralNetwork(input_dim=16)

train_dataset = tensor_gen(0)
test_dataset  = tensor_gen(1)

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True,  num_workers=0)
test_dataloader  = DataLoader(test_dataset,  batch_size=64, shuffle=False, num_workers=0)

# Compute weights inversely proportional to class frequency
y_action_all = torch.cat([batch[1] for batch in DataLoader(train_dataset, batch_size=1024)])
class_counts = torch.bincount(y_action_all.float().long())
weights = 1.0 / class_counts.float()
weights = weights / weights.sum() * len(weights)  # normalize
loss_fn_1 = nn.CrossEntropyLoss(weight=weights.to('cpu'))  # device = 'cuda' or 'cpu'
loss_fn_2 = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)






epochs = 20
for t in range(epochs):
    print(f"Epoch {t+1}/{epochs}")
    train_loop(train_dataloader, model, loss_fn_1, loss_fn_2, optimizer)
    test_loop(test_dataloader, model, loss_fn_1, loss_fn_2)

print("Done!")
torch.save(model.state_dict(), "./poker_mimic.pth")