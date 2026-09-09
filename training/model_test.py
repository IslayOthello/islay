"""Deterministic architecture, encoding, symmetry and gradient smoke checks."""
import unittest

import torch

from model import PARAMETERS, PolicyValueNet, encode, policy_value_loss, transform_action, transform_policy


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(20260909)

    def test_architecture(self):
        model = PolicyValueNet().eval()
        self.assertEqual(sum(p.numel() for p in model.parameters()), PARAMETERS)
        with torch.inference_mode():
            for batch in (1, 8, 32, 128):
                p, v = model(torch.zeros(batch, 2, 8, 8))
                self.assertEqual(p.shape, (batch, 65))
                self.assertEqual(v.shape, (batch, 1))
                self.assertTrue(torch.isfinite(p).all() and (v.abs() <= 1).all())

    def test_encoding(self):
        for sq in range(64):
            x = encode([(1 << sq, 0), (0, 1 << sq)])
            self.assertEqual(x.sum(), 2)
            self.assertEqual(x[0, 0, sq // 8, sq % 8], 1)
            self.assertEqual(x[1, 1, sq // 8, sq % 8], 1)
        for board in ((1, 1), (-1, 0), (0, 1 << 64)):
            with self.assertRaises(ValueError):
                encode([board])

    def test_symmetry(self):
        p = torch.arange(65).float()
        for symmetry in range(8):
            transformed = transform_policy(p, symmetry)
            self.assertTrue(torch.equal(transform_policy(transformed, symmetry, inverse=True), p))
            self.assertEqual(transformed[64], 64)
            for a in range(64):
                self.assertEqual(transformed[transform_action(a, symmetry)], a)

    def test_gradient_smoke(self):
        model = PolicyValueNet().train()
        inputs = encode([(1 << 28 | 1 << 35, 1 << 27 | 1 << 36), (2, 1)])
        legal = torch.zeros(2, 65, dtype=torch.bool)
        legal[0, [19, 26, 37, 44]] = True
        legal[1, 64] = True
        target = legal.float() / legal.sum(dim=1, keepdim=True)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        before = model.trunk[0][0].weight.detach().clone()
        p, v = model(inputs)
        loss = policy_value_loss(p, v, target, torch.tensor([[1.0], [-1.0]]), legal)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        optimizer.step()
        self.assertFalse(torch.equal(before, model.trunk[0][0].weight))
        with self.assertRaises(ValueError):
            policy_value_loss(p, v, target, torch.zeros(2, 1), torch.zeros_like(legal))
        with self.assertRaises(ValueError):
            policy_value_loss(p * torch.nan, v, target, torch.zeros(2, 1), legal)
        with self.assertRaises(ValueError):
            policy_value_loss(p[:0], v[:0], target[:0], v[:0], legal[:0])


if __name__ == "__main__":
    unittest.main()
