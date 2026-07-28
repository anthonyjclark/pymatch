import unittest

from match.extras import get_binary_mnist_one_batch, load_mnist_dataset


class TestExtras(unittest.TestCase):
    def test_load_mnist_dataset(self):
        train_ds, valid_ds = load_mnist_dataset()
        self.assertEqual(len(train_ds), 60000)
        self.assertEqual(len(valid_ds), 10000)
        sample_x, sample_y = train_ds[0]
        self.assertEqual(sample_x.shape, (784,))
        self.assertEqual(sample_y.shape, ())

    def test_missing_mnist_bin_raises_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            load_mnist_dataset(data_dir="/invalid/nonexistent/path")

    def test_get_binary_mnist_one_batch(self):
        X_tr, y_tr, X_va, y_va = get_binary_mnist_one_batch(classA=1, classB=7)
        self.assertGreater(len(X_tr), 0)
        self.assertEqual(len(X_tr), len(y_tr))
        self.assertGreater(len(X_va), 0)
        self.assertEqual(len(X_va), len(y_va))


if __name__ == "__main__":
    unittest.main()
