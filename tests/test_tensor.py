import unittest

import torch

import match


class TestTensor(unittest.TestCase):
    def test_tensor_init(self):
        match_t = match.tensor([1.0, 2.0, 3.0], requires_grad=True)
        torch_t = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

        self.assertEqual(match_t.shape, tuple(torch_t.shape))
        self.assertEqual(match_t.requires_grad, torch_t.requires_grad)

    def test_add_subtract_mul(self):
        match_x = match.tensor([3.0], requires_grad=True)
        match_y = match.tensor([-2.0], requires_grad=True)
        match_f = match_x * match_x * match_y + 4.0 * match_x
        match_f.backward()

        torch_x = torch.tensor([3.0], requires_grad=True)
        torch_y = torch.tensor([-2.0], requires_grad=True)
        torch_f = torch_x * torch_x * torch_y + 4.0 * torch_x
        torch_f.backward()

        self.assertAlmostEqual(match_f.item(), torch_f.item())
        self.assertAlmostEqual(match_x.grad.item(), torch_x.grad.item())
        self.assertAlmostEqual(match_y.grad.item(), torch_y.grad.item())

    def test_pow(self):
        match_x = match.tensor([2.0, 3.0, 4.0], requires_grad=True)
        match_y = match_x**3.0
        match_loss = match_y.sum()
        match_loss.backward()

        torch_x = torch.tensor([2.0, 3.0, 4.0], requires_grad=True)
        torch_y = torch_x**3.0
        torch_loss = torch_y.sum()
        torch_loss.backward()

        self.assertEqual(match_y.data.tolist(), torch_y.detach().tolist())
        self.assertEqual(match_x.grad.tolist(), torch_x.grad.tolist())

    def test_matmul_gradient(self):
        match_A = match.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        match_B = match.tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
        match_C = match_A @ match_B
        loss = match_C.sum()
        loss.backward()

        torch_A = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        torch_B = torch.tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
        torch_C = torch_A @ torch_B
        torch_loss = torch_C.sum()
        torch_loss.backward()

        self.assertEqual(match_C.data.tolist(), torch_C.detach().tolist())
        self.assertEqual(match_A.grad.tolist(), torch_A.grad.tolist())
        self.assertEqual(match_B.grad.tolist(), torch_B.grad.tolist())

    def test_activations(self):
        match_x = match.tensor([-2.0, 0.0, 3.0], requires_grad=True)
        match_y = match_x.relu()
        match_y.sum().backward()

        torch_x = torch.tensor([-2.0, 0.0, 3.0], requires_grad=True)
        torch_y = torch.relu(torch_x)
        torch_y.sum().backward()

        self.assertEqual(match_y.data.tolist(), torch_y.detach().tolist())
        self.assertEqual(match_x.grad.tolist(), torch_x.grad.tolist())

    def test_zero_grad(self):
        x = match.tensor([2.0], requires_grad=True)
        y = x * x
        y.backward()
        self.assertAlmostEqual(x.grad.item(), 4.0)

        x.zero_grad()
        self.assertAlmostEqual(x.grad.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
