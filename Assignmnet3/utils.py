# Imports:
# --------
import torch
import random
from collections import deque
import torch.nn.functional as F
import numpy as np


# Replay Buffer:
# -------------
class ReplayBuffer:
    def __init__(self, buffer_limit):
        self.buffer = deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = zip(*transitions)

        
        states      = torch.from_numpy(np.array(s_lst)).float()
        actions     = torch.tensor(a_lst, dtype=torch.long)
        rewards     = torch.tensor(r_lst, dtype=torch.float32)
        next_states = torch.from_numpy(np.array(s_prime_lst)).float()
        done_masks  = torch.tensor(done_mask_lst, dtype=torch.float32)

        return states, actions, rewards, next_states, done_masks

    def size(self):
        return len(self.buffer)


# Train function:
# ---------------
def train(q_net, 
          q_target, 
          memory, 
          optimizer,
          batch_size,
          gamma):
    
    device = next(q_net.parameters()).device 
    #! We sample from the same Replay Buffer n=10 times. You can change this of course.
    for _ in range(10):
        #! Monte Carlo sampling of a batch
        s, a, r, s_prime, done_mask = memory.sample(batch_size)

        s, a, r, s_prime, done_mask = memory.sample(batch_size)
        s         = s.to(device)
        a         = a.unsqueeze(1).to(device)          
        r         = r.unsqueeze(1).to(device)
        s_prime   = s_prime.to(device)
        done_mask = done_mask.unsqueeze(1).to(device)


        #! Get the Q-values
        q_out = q_net(s)

        #! DQN update rule
        q_a = q_out.gather(1, a)
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        target = r + gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
