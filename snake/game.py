import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np
import math

pygame.init()
pygame.mixer.quit()
font = pygame.font.Font('arial.ttf', 25)
#font = pygame.font.SysFont('arial', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)
GREEN = (0, 255, 0)

BLOCK_SIZE = 20
SPEED = 1000000

class SnakeGameAI:

    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()


    def reset(self):
        # init game state
        self.direction = Direction.RIGHT

        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head,
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]

        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0


    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()


    def play_step(self, action):
        self.frame_iteration += 1
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # 1. Calculate distance BEFORE move
        # (You can use Manhattan distance abs(x1-x2)+abs(y1-y2) or Euclidean)
        dist_before = math.sqrt((self.head.x - self.food.x)**2 + (self.head.y - self.food.y)**2)

        # 2. move
        self._move(action) # update the head
        reward = -0.1
        self.snake.insert(0, self.head)
        
        # 3. Calculate distance AFTER move
        dist_after = math.sqrt((self.head.x - self.food.x)**2 + (self.head.y - self.food.y)**2)

        # 3. check if game over
        game_over = False
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -10
            grid = self._update_ui()
            return reward, game_over, self.score, grid

        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 20
            self._place_food()
        else:
            # Snake didn't eat, so check if it moved closer
            if dist_after < dist_before:
                reward = 0.1  # Reward for getting closer
            else:
                reward = -0.2 # Penalty for moving away (optional, but helps)
                
            self.snake.pop()
        
        # 5. update ui and clock
        grid = self._update_ui()
        self.clock.tick(SPEED)
        # 6. return game over and score
        return reward, game_over, self.score, grid


    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # hits itself
        if pt in self.snake[1:-1]:
            return True

        return False


    def _update_ui(self):
        self.display.fill(BLACK)

        for pt in self.snake:
            color = GREEN if pt == self.snake[0] else BLUE1
            pygame.draw.rect(self.display, color, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()
        grid = self.get_grid_state()

        return grid

    def get_grid_state(self):
        cols = self.w // BLOCK_SIZE  # 32
        rows = self.h // BLOCK_SIZE  # 24

        # 4 binary channels: board, snake_body, snake_head, food
        board = np.ones((rows, cols), dtype=np.float32)       # 1 = empty cell
        snake_body = np.zeros((rows, cols), dtype=np.float32)
        snake_head = np.zeros((rows, cols), dtype=np.float32)
        food = np.zeros((rows, cols), dtype=np.float32)

        # food channel
        fx = int(self.food.x // BLOCK_SIZE)
        fy = int(self.food.y // BLOCK_SIZE)
        if 0 <= fx < cols and 0 <= fy < rows:
            food[fy, fx] = 1.0
            board[fy, fx] = 0.0

        # snake body + head channels
        for i, pt in enumerate(self.snake):
            x = int(pt.x // BLOCK_SIZE)
            y = int(pt.y // BLOCK_SIZE)
            if 0 <= x < cols and 0 <= y < rows:
                if i == 0:
                    snake_head[y, x] = 1.0
                else:
                    snake_body[y, x] = 1.0
                board[y, x] = 0.0

        grid = np.stack([board, snake_body, snake_head, food], axis=0)  # (4, 24, 32)
        assert grid.shape == (4, 24, 32), f"grid shape is {grid.shape}, expected (4, 24, 32)"
        return grid  # shape (4, 24, 32)


    def _move(self, action):
        # [straight, right, left]

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx] # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # right turn r -> d -> l -> u
        else: # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # left turn r -> u -> l -> d

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)