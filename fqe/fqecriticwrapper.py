import torch
from fqe.critic import Critic

class FQECriticWrapper:
    def __init__(self):
        self.fqe = Critic.load("1000000", model_dir="fqe/")

    def predict(self, states, actions):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return self.fqe.predict(states, actions, device=device)