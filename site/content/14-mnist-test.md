# MNIST Test Page

This test page verifies loading the preprocessed MNIST dataset with PyMatch and plots a random sample of 16 images in an 8&times;2 grid layout using **Matplotlib**.

## Interactive MNIST Sample Plotter

Click **Run** below to load MNIST using PyMatch and plot a random 8&times;2 grid of 16 samples with `matplotlib.pyplot`:

```python
import match
from match.extras import load_mnist_dataset
import matplotlib.pyplot as plt
import random

# 1. Load MNIST dataset partitions
train_ds, valid_ds = load_mnist_dataset()

# 2. Select 16 random sample indices (8x2 grid)
indices = random.sample(range(len(train_ds)), 16)

# 3. Create 8x2 subplot grid using Matplotlib
fig, axes = plt.subplots(2, 8, figsize=(12, 3.5))

for i, idx in enumerate(indices):
    img_tensor, label = train_ds[idx]
    
    # Reshape (784,) 1D tensor data to (28, 28) 2D image matrix
    img_data = img_tensor.data.data
    grid_2d = [img_data[r * 28 : (r + 1) * 28] for r in range(28)]
    
    ax = axes[i // 8, i % 8]
    ax.imshow(grid_2d, cmap="gray")
    ax.set_title(f"Label: {int(label.item())}", fontsize=9)
    ax.axis("off")

plt.tight_layout()
plt.show()

print(f"Loaded {len(train_ds)} train samples and {len(valid_ds)} validation samples.")
```

<div class="interactive-playground">
  <h3>MNIST Matplotlib Sample Plotter</h3>
  <textarea id="code-input" rows="22">import match
from match.extras import load_mnist_dataset
import matplotlib.pyplot as plt
import random

# Load dataset partitions
train_ds, valid_ds = load_mnist_dataset()

# Select 16 random sample indices (8x2 grid)
indices = random.sample(range(len(train_ds)), 16)

# Create 8x2 subplot grid
fig, axes = plt.subplots(2, 8, figsize=(12, 3.5))

for i, idx in enumerate(indices):
    img_tensor, label = train_ds[idx]
    
    img_data = img_tensor.data.data
    grid_2d = [img_data[r * 28 : (r + 1) * 28] for r in range(28)]
    
    ax = axes[i // 8, i % 8]
    ax.imshow(grid_2d, cmap="gray")
    ax.set_title(f"Label: {int(label.item())}", fontsize=9)
    ax.axis("off")

plt.tight_layout()
plt.show()

print(f"Loaded {len(train_ds)} train and {len(valid_ds)} valid samples.")
</textarea>
  <button id="run-btn">Run</button>
  <pre id="output-text"></pre>
</div>
