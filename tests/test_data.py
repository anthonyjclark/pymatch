import unittest

import match
from match.utils.data import DataLoader, Dataset, TensorDataset


class CustomDataset(Dataset):
    def __init__(self, size: int = 10):
        self.size = size
        self.data = match.randn(size, 4)
        self.targets = match.zeros(size)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int):
        return self.data[index], self.targets[index]


class TestData(unittest.TestCase):
    def test_tensor_dataset_single(self):
        X = match.randn(20, 5)
        ds = TensorDataset(X)
        self.assertEqual(len(ds), 20)
        sample = ds[0]
        self.assertEqual(sample.shape, (5,))

    def test_tensor_dataset_multiple(self):
        X = match.randn(20, 5)
        y = match.zeros(20)
        ds = TensorDataset(X, y)
        self.assertEqual(len(ds), 20)
        sample_x, sample_y = ds[0]
        self.assertEqual(sample_x.shape, (5,))

    def test_dataloader_batching(self):
        X = match.randn(10, 4)
        y = match.zeros(10)
        ds = TensorDataset(X, y)
        loader = DataLoader(ds, batch_size=3, shuffle=False)

        self.assertEqual(len(loader), 4)

        batches = list(loader)
        self.assertEqual(len(batches), 4)

        # Batch 0..2 should have size 3
        batch_0_x, batch_0_y = batches[0]
        self.assertEqual(batch_0_x.shape, (3, 4))
        self.assertEqual(batch_0_y.shape, (3,))

        # Last batch should have leftover size 1
        batch_3_x, batch_3_y = batches[3]
        self.assertEqual(batch_3_x.shape, (1, 4))
        self.assertEqual(batch_3_y.shape, (1,))

    def test_dataloader_drop_last(self):
        X = match.randn(10, 4)
        loader = DataLoader(X, batch_size=3, drop_last=True)
        self.assertEqual(len(loader), 3)

        batches = list(loader)
        self.assertEqual(len(batches), 3)
        for batch_x in batches:
            self.assertEqual(batch_x.shape, (3, 4))

    def test_dataloader_shuffle(self):
        match.manual_seed(42)
        X = match.randn(50, 4)
        loader_unshuffled = DataLoader(X, batch_size=10, shuffle=False)
        loader_shuffled = DataLoader(X, batch_size=10, shuffle=True)

        batch_unshuffled = next(iter(loader_unshuffled))
        batch_shuffled = next(iter(loader_shuffled))

        # Shapes must match
        self.assertEqual(batch_unshuffled.shape, (10, 4))
        self.assertEqual(batch_shuffled.shape, (10, 4))

    def test_custom_dataset(self):
        ds = CustomDataset(size=12)
        loader = DataLoader(ds, batch_size=4)
        self.assertEqual(len(loader), 3)

        for batch_x, batch_y in loader:
            self.assertEqual(batch_x.shape, (4, 4))
            self.assertEqual(batch_y.shape, (4,))


if __name__ == "__main__":
    unittest.main()
