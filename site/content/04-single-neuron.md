# 4. Single Neuron

> A single neuron in the brain is an incredibly complex machine that even today we don’t understand. A single “neuron” in a neural network is an incredibly simple mathematical function that captures a minuscule fraction of the complexity of a biological neuron.
>
> — [Andrew Ng](https://www.wired.com/2015/02/google-brains-co-inventor-tells-why-hes-building-chinese-neural-networks/)

When our model is a single neuron we can only produce a single output. So, $n_y=1$ for this section. Sticking to our MNIST digits example from above, we could train a single neuron to distinguish between two different classes of digits (e.g., "1" vs "7", "0" vs "non-zero", etc.).

## Notation and Diagram

Here is a diagram representing a single neuron (as we'll see later, some neural networks are just many of these neurons interconnected):

![A neuron model with separate nodes for linear and activation computations.](/img/NeuronSeparate.svg)

The diagram represents the following equations:

$$
\begin{align}
z^{(i)} &= \sum_{k=1}^{n_x} x_k^{(i)} w_k + b\\
a^{(i)} &= g(z^{(i)})
\end{align}
$$

For these two equations:

- $x_k^{(i)}$ are the input features for the $i^{th}$ example
- $w_k$ (weights) and $b$ (bias) are the **learned** parameters
- $z^{(i)}$ is a weighted sum of the input features plus the additional bias term
- $a^{(i)}$ is the output of a non-linear activation function $g(\mathord{\cdot})$ applied to $z^{(i)}$
- $\yhat^{(i)}$ is the label we often give to the output ($a^{(i)} = \yhat^{(i)}$)

<details class="question">
<summary><strong>Question:</strong> Why do $w_k$ and $b$ not have superscripts?</summary>
<div class="answer">
<strong>Answer:</strong> The parameters $w_k$ and $b$ do not change as the input $x_k^{(i)}$ changes. These parameters <strong>are</strong> the neuron, and they are used to produce the output $\yhat^{(i)}$ for any given input; we use the same parameter values regardless of input.
</div>
</details>

**For this model, we want to find parameters $w_k$ and $b$ such that the neuron outputs $\yhat^{(i)} \approx y^{(i)}$ for any input.**

Below is a more common representation of a neuron model, combining the linear and activation components:

![A neuron model.](/img/Neuron.svg)

## Neuron with Python Standard Libraries

Here is a simple python implementation of a neuron without optimization:

```python
from math import exp
from random import gauss

def sigmoid(z: float) -> float:
    """The sigmoid/logistic activation function."""
    return 1 / (1 + exp(-z))

N = 100
nx = 4
x1 = [gauss(0, 1) for _ in range(N)]
x2 = [gauss(0, 1) for _ in range(N)]
x3 = [gauss(0, 1) for _ in range(N)]
x4 = [gauss(0, 1) for _ in range(N)]

w1 = gauss(0, 1)
w2 = gauss(0, 1)
w3 = gauss(0, 1)
w4 = gauss(0, 1)
b = 0

for x1i, x2i, x3i, x4i in zip(x1, x2, x3, x4):
    zi = w1 * x1i + w2 * x2i + w3 * x3i + w4 * x4i + b
    ai = sigmoid(zi)
```

In this code listing I use the `sigmoid` activation function. This function is plotted below.

![Sigmoid activation function and its derivative.](/img/Sigmoid.png)

Some nice properties of this function include:

- An output range of [0, 1] (all inputs are "squashed" into this range).
- An easy to compute derivative.
- Easy to interpret and understand.

We often use sigmoid activation functions for binary classification (predicting whether an input belongs to one of two classes).

<details class="question">
<summary><strong>Question:</strong> Can you think of any downsides for this function (hint: look at the derivative curve)?</summary>
<div class="answer">
<strong>Answer:</strong> While this function was once widely used, it can lead to slower learning due to small derivative values for inputs $z$ outside the range [-4, 4] (vanishing gradients). ReLU is more commonly used for hidden layers today.
</div>
</details>

## The Dot-Product

We compute $z^{(i)}$ using a summation, but we can express this same bit of math using the dot-product from linear algebra:

$$
z^{(i)} = \sum_{k=1}^{n_x} x_k^{(i)} w_k + b = \vx^{(i)T} \vw + b
$$

Using PyMatch (`match`):

```python
import match

N = 100
nx = 4
X = match.randn(N, nx)
w = match.randn(nx)
b = 0

for xi in X:
    zi = xi @ w + b
    ai = zi.sigmoid()
```

## Vectorizing Inputs

In addition to using a dot-product, we can use matrix multiplication in place of looping over all examples in the dataset:

$$
\begin{align}
\vz &= X \vw + \mathbf{1} b \\
\va &= g(\vz)
\end{align}
$$

```python
import match

N = 100
nx = 4
X = match.randn(N, nx)
w = match.randn(nx)
b = 0

z = X @ w + b
yhat = z.sigmoid()
```

<details class="question">
<summary><strong>Question:</strong> What are the dimensions of $\vz$ and $\va$ (aka $\vyhat$)?</summary>
<div class="answer">
<strong>Answer:</strong> We are computing a single output value for each input, so the shape of these vectors is $(N \times 1)$. PyMatch will treat these as arrays with $N$ elements instead of as column vectors.
</div>
</details>

## Optimization with Batch Gradient Descent

We must find values for parameters $\vw$ and $b$ to make $\yhat^{(i)} \approx y^{(i)}$. We use gradient descent to optimize parameters.

The standard choice when performing classification with a neuron is **binary cross-entropy** (BCE):

$$
\begin{align}
ℒ(\vyhat, \vy) &= -\frac{1}{N}\sum_{i=1}^N (y^{(i)} \log{\yhat^{(i)}} + (1 - y^{(i)})\log{(1-\yhat^{(i)})})\\
  &= -\text{mean}_0\left(\vy \cdot \log{\vyhat} + (1 - \vy) \cdot \log{(1 - \vyhat)}\right)
\end{align}
$$

![The effect on loss ℒ of adjusting parameter w_k.](/img/LossLandscape.svg)

Using the chain rule, partial derivatives with respect to parameters are:

$$
\begin{align}
\frac{∂ ℒ}{∂ \vw} &= \frac{1}{N} X^T (\vyhat - \vy)\\[10pt]
\frac{∂ ℒ}{∂ b} &= \frac{1}{N} \sum_{i=1}^N (\yhat^{(i)} - y^{(i)})
\end{align}
$$

Parameter updates:

$$
\begin{align}
\vw &:= \vw - η \frac{∂ ℒ}{∂ \vw} \\
b &:= b - η \frac{∂ ℒ}{∂ b}
\end{align}
$$

## Neuron Batch Gradient Descent Code

Here is a complete example training a neuron using PyMatch (`match`) and `match.extras` to classify binary MNIST digits:

```python
import match
from match.extras import get_binary_mnist_one_batch

# Load binary MNIST dataset (digit 1 vs digit 7)
train_X, train_y, valid_X, valid_y = get_binary_mnist_one_batch("../Data", classA=1, classB=7, flatten=True)

# Neuron parameters
nx = 28 * 28
w = match.randn(nx) * 0.01
b = match.zeros(1)

num_epochs = 4
learning_rate = 0.01

# Training loop
for epoch in range(num_epochs):
    # Forward pass
    yhat = (train_X @ w + b).sigmoid()
    losses = -(train_y * yhat.log() + (1 - train_y) * (1 - yhat).log())

    # Backward pass (gradients)
    dz = yhat - train_y
    dw = (1 / train_y.shape[0]) * (dz @ train_X)
    db = dz.mean()

    # Update parameters
    w -= learning_rate * dw
    b -= learning_rate * db
```
