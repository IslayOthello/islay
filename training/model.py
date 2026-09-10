"""Configuration B: shared 8x64 residual policy/value network, encoding v1."""
import torch
from torch import nn

ARCHITECTURE = "b8c64-v1"
ENCODING = "relative-2x8x8-a1-v1"
POLICY = "a1-h8-pass65-v1"
PARAMETERS = 605960


class ConvBlock(nn.Sequential):
    def __init__(self, inputs, outputs, kernel):
        super().__init__(nn.Conv2d(inputs, outputs, kernel, padding=kernel // 2, bias=False),
                         nn.BatchNorm2d(outputs, eps=1e-5, momentum=0.1), nn.ReLU())


class ResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = ConvBlock(64, 64, 3)
        self.second = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(64, eps=1e-5, momentum=0.1))

    def forward(self, x):
        return torch.relu(x + self.second(self.first(x)))


class PolicyValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = nn.Sequential(ConvBlock(2, 64, 3), *(ResidualBlock() for _ in range(8)))
        self.policy = nn.Sequential(ConvBlock(64, 2, 1), nn.Flatten(), nn.Linear(128, 65))
        self.value = nn.Sequential(ConvBlock(64, 1, 1), nn.Flatten(), nn.Linear(64, 64),
                                   nn.ReLU(), nn.Linear(64, 1), nn.Tanh())

    def forward(self, board):
        features = self.trunk(board)
        return self.policy(features), self.value(features)


def encode(boards):
    """Mover-relative uint64 pairs; row 0 = rank 1, column 0 = file a."""
    planes = []
    for player, opponent in boards:
        if min(player, opponent) < 0 or max(player, opponent) >= 1 << 64 or player & opponent:
            raise ValueError("expected disjoint uint64 bitboards")
        planes.append([[(bits >> sq) & 1 for sq in range(64)] for bits in (player, opponent)])
    return torch.tensor(planes, dtype=torch.float32).reshape(len(boards), 2, 8, 8)


def transform_action(action, symmetry):
    if not 0 <= action <= 64 or not 0 <= symmetry < 8:
        raise ValueError("invalid action or symmetry")
    if action == 64:
        return action
    x, y = action % 8, action // 8
    if symmetry & 1:
        x = 7 - x
    if symmetry & 2:
        y = 7 - y
    if symmetry & 4:
        x, y = y, x
    return y * 8 + x


def transform_policy(policy, symmetry, inverse=False):
    """Move policy/mask with the board; inverse maps predictions to the original board."""
    mapping = torch.tensor([transform_action(a, symmetry) for a in range(65)], device=policy.device)
    if inverse:
        return policy[..., mapping]
    result = torch.empty_like(policy)
    result[..., mapping] = policy
    return result


def policy_value_loss(logits, value, target_policy, outcome, legal):
    """Terminal samples are excluded. Regularization belongs to the optimizer/config."""
    if (logits.ndim != 2 or logits.shape[0] == 0 or logits.shape != target_policy.shape
            or logits.shape != legal.shape or logits.shape[-1] != 65):
        raise ValueError("policy shape mismatch")
    if value.shape != outcome.shape or value.shape != (logits.shape[0], 1):
        raise ValueError("value shape mismatch")
    if not torch.isfinite(logits).all() or not torch.isfinite(value).all():
        raise ValueError("non-finite network output")
    legal = legal.bool()
    if not legal.any(dim=1).all() or not torch.isfinite(target_policy).all() or (target_policy < 0).any():
        raise ValueError("invalid policy target/mask")
    if (target_policy.masked_select(~legal) != 0).any() or not torch.allclose(
            target_policy.sum(dim=1), torch.ones_like(target_policy[:, 0]), atol=1e-6):
        raise ValueError("target policy must sum to one over legal actions")
    if not torch.isfinite(outcome).all() or (outcome.abs() > 1).any():
        raise ValueError("invalid outcome")
    policy, mse, _ = policy_value_terms(logits, value, target_policy, outcome, legal)
    return policy.mean() + mse.mean()


def policy_value_terms(logits, value, target_policy, outcome, legal):
    """Per-position terms for already validated batches; avoid per-field GPU synchronization."""
    logp = torch.log_softmax(logits.masked_fill(~legal, -torch.inf), dim=1).masked_fill(~legal, 0)
    probabilities = logp.exp().masked_fill(~legal, 0)
    return (-(target_policy * logp).sum(dim=1), (value - outcome).square().flatten(),
            -(probabilities * logp).sum(dim=1))
