import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import DQN_CNN, QTrainer
from helper import plot
from collections import deque
import os

MAX_MEMORY = 100_000
BATCH_SIZE = 256
LR = 0.001
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = DQN_CNN(in_channels = 16, num_actions=3).to(device)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

        # load weights if they exist
        model_path = "./model/model.pth"
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location="cpu")
            self.model.load_state_dict(state_dict)
            print("Loaded model weights from", model_path)

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)

        states = np.stack(states, axis=0)         # (B, 4, 24, 32)
        next_states = np.stack(next_states, axis=0)
        actions = np.array(actions, dtype=float)  # (B, 3)
        rewards = np.array(rewards, dtype=float)  # (B,)
        dones = np.array(dones, dtype=bool)       # (B,)

        loss = self.trainer.train_step(states, actions, rewards, next_states, dones)
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)
        return loss
    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        #Convert state stack to tensor
        state = torch.from_numpy(state).unsqueeze(0).float().to(device)
        # random moves: tradeoff exploration / exploitation
        self.epsilon = max(5, 300 - self.n_games)
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
        else:
            #if state.dim() == 3:  # (4,H,W)
                #state = state.unsqueeze(0)
            with torch.no_grad():
                prediction = self.model(state)
            move = torch.argmax(prediction, dim=1).item()

        final_move[int(move)] = 1
        return final_move
           


class FrameStack:
    def __init__(self, k: int):
        self.k = k
        self.frames = deque(maxlen=k)

    def reset(self, frame_84x84: np.ndarray) -> np.ndarray:
        """Fill the stack with the first frame repeated k times."""
        self.frames.clear()
        for _ in range(self.k):
            self.frames.append(frame_84x84)
        return self._stack()

    def step(self, frame_84x84: np.ndarray) -> np.ndarray:
        """Add a new frame and return stacked state."""
        self.frames.append(frame_84x84)
        return self._stack()

    def _stack(self) -> np.ndarray:
        # Each frame is (4, 24, 32); concatenate along channel axis -> (k*4, 24, 32)
        arr = np.concatenate(self.frames, axis=0)
        assert arr.shape == (self.k * 4, 24, 32), f"stacked frame shape {arr.shape}"
        return arr

def train():
    plot_scores = []
    plot_mean_scores = []
    losses = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    grid_stack = FrameStack(k = 4)
    grid = game.get_grid_state()
    state = grid_stack.reset(grid)
    while True:

        #Flatten grids
        #
        # get move
        final_move = agent.get_action(state)

        # perform move and get new state
        reward, done, score, grid = game.play_step(final_move)
        state_new = grid_stack.step(grid) 
        #Flatten frame into tensor
        #state_new = torch.from_numpy(state_new).unsqueeze(0).float()

        # train short memory
        agent.train_short_memory(state, final_move, reward, state_new, done)

        # remember
        agent.remember(state, final_move, reward, state_new, done)

        state = state_new
        if done:
            episode_score = score
            # train long memory, plot result
            game.reset()
            grid = game.get_grid_state()
            state = grid_stack.reset(grid)

            agent.n_games += 1
            loss = agent.train_long_memory()
            if loss is not None:
                losses.append(loss)

            if episode_score > record:
                record = episode_score
                agent.model.save()
            count = 0
            if count % 100 == 0:
                with torch.no_grad():
                    s = torch.from_numpy(state).unsqueeze(0).float().to(device)
                    q = agent.model(s)
                    print("Q:", q.detach().cpu().numpy())
            count += 1

            print('Game', agent.n_games, 'Score', episode_score, 'Record:', record)

            plot_scores.append(episode_score)
            total_score += episode_score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)

            save_path = None
            if agent.n_games % 100 == 0:
                os.makedirs("./figure", exist_ok=True)
                save_path = f"./figure/figure.png"
            

            plot(plot_scores, plot_mean_scores, losses, save_path =save_path)



    
if __name__ == '__main__':
    train()