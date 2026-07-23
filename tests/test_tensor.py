import unittest
import match


class TestTensor(unittest.TestCase):
    def test_tensor_init(self):
        t1 = match.Tensor([1.0, 2.0, 3.0], requires_grad=True)
        self.assertEqual(t1.shape, (3,))
        self.assertTrue(t1.requires_grad)
        self.assertIsNotNone(t1.grad)
        self.assertEqual(t1.grad.tolist(), [0.0, 0.0, 0.0])

    def test_add_subtract_mul(self):
        x = match.tensor([3.0], requires_grad=True)
        y = match.tensor([-2.0], requires_grad=True)

        f = x * x * y + 4.0 * x
        f.backward()

        self.assertAlmostEqual(x.grad.item(), -8.0)
        self.assertAlmostEqual(y.grad.item(), 9.0)

    def test_matmul_gradient(self):
        match.manual_seed(42)
        A = match.Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        B = match.Tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)

        C = A @ B
        loss = C.sum()
        loss.backward()

        expected_dA = [[11.0, 15.0], [11.0, 15.0]]
        expected_dB = [[4.0, 4.0], [6.0, 6.0]]

        self.assertEqual(A.grad.tolist(), expected_dA)
        self.assertEqual(B.grad.tolist(), expected_dB)

    def test_activations(self):
        x = match.Tensor([-2.0, 0.0, 3.0], requires_grad=True)
        y = x.relu()
        y.sum().backward()

        self.assertEqual(y.data.tolist(), [0.0, 0.0, 3.0])
        self.assertEqual(x.grad.tolist(), [0.0, 0.0, 1.0])

    def test_zero_grad(self):
        x = match.tensor([2.0], requires_grad=True)
        y = x * x
        y.backward()
        self.assertAlmostEqual(x.grad.item(), 4.0)

        x.zero_grad()
        self.assertAlmostEqual(x.grad.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
