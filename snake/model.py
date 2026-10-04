import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

#Input size should be stack of frames
class DQN_CNN(nn.Module):
    def __init__(self, in_channels: int, num_actions: int):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        # compute flatten size with a dummy forward pass (robust)
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 24, 32)
            n_flat = self.features(dummy).view(1, -1).size(1)

        self.head = nn.Sequential(
            nn.Linear(n_flat, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        # x: (B, C, H, W) float32, binary channels [0,1]
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.head(x)

    def save(self, file_name="model.pth"):
        os.makedirs("./model", exist_ok=True)
        torch.save(self.state_dict(), os.path.join("./model", file_name))


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.SmoothL1Loss()

    def train_step(self, state, action, reward, next_state, done):
        state = np.array(state)
        next_state = np.array(next_state)

        state = torch.from_numpy(state).float().to(device)       # (B, C, H, W) or (C, H, W)
        next_state = torch.from_numpy(next_state).float().to(device)
        action = torch.tensor(action, dtype=torch.float).to(device)
        reward = torch.tensor(reward, dtype=torch.float).to(device)
        # (n, x)

        #If single sample add batch dim
        if len(state.shape) == 3:
            # (1, x)
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            done = (done, )

        # 1: predicted Q values with current state
        pred = self.model(state)

        with torch.no_grad():
            pred_next = self.model(next_state)
        target = pred.clone().detach()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(pred_next[idx])

            target[idx][torch.argmax(action[idx]).item()] = Q_new
    
        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()
        # debug
        total_norm = 0.0
        #for p in self.model.parameters():
            #if p.grad is not None:
               # total_norm += p.grad.data.norm(2).item()
        #print("grad norm:", total_norm)
        
        if np.random.rand() < 0.001:
            print("Loss:", loss.item())
        return loss.item()
        



