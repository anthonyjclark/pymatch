# 5. Neural Networks and Backpropagation

> Once your computer is pretending to be a neural net, you get it to be able to do a particular task by just showing it a whole lot of examples.
>
> — Geoffrey Hinton

Below is our first neural network (aka multi-layer perceptron, MLP). We'll start by using this diagram to formulate terminology and conventions:

![A two-layer neural network.](/img/2LayerNetwork.svg)

Notation:

- Layer 0 is the input (we called this $X$ for a single Neuron)
- Square bracket superscripts denote the network layer
- Round parenthesis superscripts denote the example index
- $w$ parameter subscripts denote first the associated neuron in the current layer and second the associated neuron from the previous layer
- $b$, $z$, and $a$ subscripts denote an associated neuron

<details class="question">
<summary><strong>Question:</strong> Given a hypothetical deep neural network, how would you denote the linear computation of the third neuron in the fifth layer for training example 6123?</summary>
<div class="answer">
<strong>Answer:</strong> $$z_3^{[5](6123)}$$
<ul>
  <li>"$z$": linear computation</li>
  <li>"$[5]$" superscript: fifth layer</li>
  <li>"$(6123)$" superscript: example 6123</li>
  <li>"$3$" subscript: third neuron</li>
</ul>
</div>
</details>

## Vectorized Equations For a Neural Network

Parameters for any layer $l = 1, 2, \dots, L$:

\begin{align}
W^{[l]} &= \begin{bmatrix} w^{[l]}_{1,1} & \cdots & w^{[l]}_{1,n_{l-1}} \\ \vdots & \ddots & \vdots \\ w^{[l]}_{n_l,1} & \cdots & w^{[l]}_{n_l,n_{l-1}} \end{bmatrix} \\
\vb^{[l]} &= \begin{bmatrix} b^{[l]}_1 \\ \vdots \\ b^{[l]}_{n_l} \end{bmatrix}
\end{align}

Vectorized linear and activation equations for each layer across all examples:

\begin{align}
Z^{[l]} &= A^{[l-1]} W^{[l]T} + \mathbf{1} \vb^{[l]T}\\
A^{[l]} &= g^{[l]}(Z^{[l]})
\end{align}

<details class="question">
<summary><strong>Question:</strong> Why do we have $\mathbf{1} \vb^{[l]T}$?</summary>
<div class="answer">
<strong>Answer:</strong> This ensures that the dimensions are correct between added matrices. Try this out in PyMatch:
<pre><code>import match
N, nl = 10, 4
b = match.randn(nl, 1)
ONE = match.ones(N, 1)
print(ONE @ b.T)
</code></pre>
</div>
</details>

<details class="question">
<summary><strong>Question:</strong> What is the shape of $Z^{[l]}$?</summary>
<div class="answer">
<strong>Answer:</strong> $Z^{[l]}$ is $(N \times n_l)$.
</div>
</details>

<details class="question">
<summary><strong>Question:</strong> What is the shape of $A^{[l]}$?</summary>
<div class="answer">
<strong>Answer:</strong> $A^{[l]}$ is $(N \times n_l)$.
</div>
</details>

## Backpropagation

Just like for the single neuron, we want to find values for $W^{[l]}$ and $\vb^{[l]}$ (for $l = 1, 2, \dots, L$) such that $A^{[L]} \approx Y$.

This process of computing derivatives backward through the network is referred to as **backpropagation**. A compute graph depicts the flow of activations (forward pass) and gradients (backward pass):

![Compute graph for two-layer network.](/img/ComputeGraph.svg)

### Layer 2 Parameters

Partial derivatives for layer 2:

\begin{align}
\frac{∂ℒ}{∂ W^{[2]}} &= \frac{1}{N} ∂_{Z^{[2]}}^T A^{[1]}\\
\frac{∂ ℒ}{∂ \vb^{[2]}} &= \text{mean}_0 (∂_{\vz^{[2]}})
\end{align}

where $∂_{Z^{[2]}} = A^{[2]} - Y$.

### Layer 1 Parameters

Partial derivatives for layer 1:

\begin{align}
\frac{∂ℒ}{∂ W^{[1]}} &= \frac{1}{N} ∂_{Z^{[1]}}^T A^{[0]}\\
\frac{∂ ℒ}{∂ \vb^{[1]}} &= \text{mean}_0 (∂_{\vz^{[1](i)}})
\end{align}

where $∂_{Z^{[1]}} = ∂_{Z^{[2]}} W^{[2]} \cdot A^{[1]} \cdot (1 - A^{[1]})$.

### Parameter Update Equations

\begin{align}
W^{[1]} &:= W^{[1]} - η \frac{∂ℒ}{∂ W^{[1]}} \\
\vb^{[1]} &:= \vb^{[1]} - η \frac{∂ℒ}{∂ \vb^{[1]}} \\
W^{[2]} &:= W^{[2]} - η \frac{∂ℒ}{∂ W^{[2]}} \\
\vb^{[2]} &:= \vb^{[2]} - η \frac{∂ℒ}{∂ \vb^{[2]}}
\end{align}

## Two-Layer Neural Network Code

```python
import match

n0 = 28 * 28
n1 = 2
n2 = 1

W1 = match.randn(n1, n0)
b1 = match.randn(n1)
W2 = match.randn(n2, n1)
b2 = match.randn(n2)

def model(A0):
    Z1 = A0 @ W1.T + b1
    A1 = Z1.sigmoid()
    Z2 = A1 @ W2.T + b2
    A2 = Z2.sigmoid()
    return Z1, A1, Z2, A2.squeeze()
```

## Automatic Differentiation

Instead of computing derivatives by hand, machine learning frameworks use **automatic differentiation**. PyMatch (`match`) has automatic differentiation built-in:

1. Creates a compute graph from your tensor operations.
2. Performs a topological sort on the compute graph.
3. Computes gradients and backpropagates them to all matrices.

```python
import match

N, n0, n1, n2 = 20, 10, 7, 13

A0 = match.randn(N, n0)
Y = match.randn(N, n2)

W1 = match.randn(n1, n0, requires_grad=True)
b1 = match.randn(n1, requires_grad=True)
Z1 = A0 @ W1.T + b1
A1 = Z1.sigmoid()

W2 = match.randn(n2, n1, requires_grad=True)
b2 = match.randn(n2, requires_grad=True)
Z2 = A1 @ W2.T + b2
A2 = Z2.sigmoid()

loss = ((A2 - Y) ** 2).mean()
loss.backward()

print("W1 gradient:", W1.grad.shape)
print("W2 gradient:", W2.grad.shape)
```

![Compute graph for two-layer network.](/img/AutoDiffComputeGraph.svg)
