# 3. Data

> Tidy datasets are all alike but every messy dataset is messy in its own way.
>
> — [Hadley Wickham](https://www.jstatsoft.org/article/view/v059i10/)

Perhaps the most important aspect of a neural network is the dataset. Let

$$\mathcal{D} = \{X, Y\}$$

denote a dataset comprising input *features* $X$ and output *targets* $Y$. Although $X$ and $Y$ can come in many shapes, I am going to be opinionated here and use a specific (and consistent) convention. Let's use $N$ to denote the size of the paired dataset. (Note, not all problems have output targets, but herein I am talking about supervised learning unless otherwise specified.)

We will frequently take a dataset and split it into examples used for training, validation, and evaluation. We'll discuss these terms near the end of this section.

$X$ is a matrix (indicated by capitalization) containing all features of all input examples. A single input example $\vx\i$ is often represented as a *column* vector (indicated by boldface):

$$\vx\i = \begin{bmatrix} x\i_{1} \\ \vdots \\ x\i_{n_x} \end{bmatrix}$$

where subscripts denote the feature index, $n_x$ is the number of features, and the superscript $i$ denotes that this is the $i^{\mathit{th}}$ training example. We do not always put the input features into a column vector, but it is fairly standard.

Each row in $X$ is a single input example (also referred to as an instance or sample), and when you stack all $N$ examples on top of each other (first transposing them into row vectors), you end up with:

$$X = \begin{bmatrix} \text{--- } \vx^{(1)T} \text{ ---} \\ \vdots \\ \text{--- } \vx^{(N)T} \text{ ---} \end{bmatrix} = \begin{bmatrix} x^{(1)}_{1} & \cdots & x^{(1)}_{n_x} \\ \vdots & \ddots & \vdots \\ x^{(N)}_{1} & \cdots & x^{(N)}_{n_x} \end{bmatrix}$$

We transpose each example column vector (i.e., $\vx^{(i)T}$) into a row vector so that the first dimension of $X$ corresponds to the number of examples $N$ and the second dimension is the number of features $n_x$. Compare the column vector above to each row in the matrix.

Let's denote matrix dimensions with $(r \times c)$ (the number of rows $r$ by the number of columns $c$ in the matrix). I will, in text and in code, refer to matrix dimensions as the "shape" of the matrix.

<details class="question">
<summary><strong>Question:</strong> What is the shape of $X$?</summary>
<div class="answer">
<strong>Answer:</strong> We say that $\vx\i \in \mathcal{R}^{n_x}$ (each input example is $n_x$ real values) and $X \in \mathcal{R}^{N \times n_x}$. Therefore, the shape of $X$ is $(N \times n_x)$.
</div>
</details>

$Y$ contains the targets (also referred to as labels or the true/correct/actual/expected output values). Here is a single target column vector:

$$\vy\i = \begin{bmatrix} y\i_{1} \\ \vdots \\ y\i_{n_y} \end{bmatrix}$$

And here is the entire target matrix including all examples:

$$Y = \begin{bmatrix} \text{--- } \vy^{(1)T} \text{ ---} \\ \vdots \\ \text{--- } \vy^{(N)T} \text{ ---} \end{bmatrix} = \begin{bmatrix} y^{(1)}_{1} & \cdots & y^{(1)}_{n_y} \\ \vdots & \ddots & \vdots \\ y^{(N)}_{1} & \cdots & y^{(N)}_{n_y} \end{bmatrix}$$

<details class="question">
<summary><strong>Question:</strong> What is the shape of $Y$?</summary>
<div class="answer">
<strong>Answer:</strong> The shape of $Y$ is $(N \times n_y)$.
</div>
</details>

Let's use the [MNIST dataset](https://en.wikipedia.org/wiki/MNIST_database) as an example. This dataset comprises a training partition including 60,000 images and a validation partition including 10,000 images. Each image is 28 pixels in height and 28 pixels in width for a total of 784 pixels. Each image depicts a single handwritten digit—a number in the range zero through nine. Here is a small sample of these images:

![MNIST Sample. Image from Wikipedia.](https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png)

<details class="question">
<summary><strong>Question:</strong> What is the shape of the training partition of the input $X_{train}$?</summary>
<div class="answer">
<strong>Answer:</strong> $X_{train}$ is $(60000 \times 784)$:
$$X = \begin{bmatrix} x^{(1)}_{1} & \cdots & x^{(1)}_{784} \\ \vdots & \ddots & \vdots \\ x^{(60000)}_{1} & \cdots & x^{(60000)}_{784} \end{bmatrix}$$
The first row includes all 784 pixels of the first training image, and subsequent rows likewise contain pixel data for a single image.
</div>
</details>

<details class="question">
<summary><strong>Question:</strong> What is the shape of the training partition of the targets $Y_{train}$?</summary>
<div class="answer">
<strong>Answer:</strong> $Y_{train}$ is $(60000 \times 10)$:
$$Y = \begin{bmatrix} y^{(1)}_{1} & \cdots & y^{(1)}_{10} \\ \vdots & \ddots & \vdots \\ y^{(60000)}_{1} & \cdots & y^{(60000)}_{10} \end{bmatrix}$$
Each row in this matrix is one-hot encoded, meaning that only one item in each row is "1" and all other items in a row are "0". Here is an example of a one-hot encoding target for an input image representing the digit "2":
$$y^T = \begin{bmatrix} 0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0\end{bmatrix}$$
For efficiency's sake, we often represent a one-hot encoded vector using just the index of the "hot" item. For example, the previous vector can be represented by the integer 2.
</div>
</details>

<details class="question">
<summary><strong>Question:</strong> What are the shapes of $X_{valid}$ and $Y_{valid}$?</summary>
<div class="answer">
<strong>Answer:</strong> $X_{valid}$ and $Y_{valid}$ are $(10000 \times 784)$ and $(10000 \times 10)$, respectively.
</div>
</details>

You might now wonder why we split a dataset into training/validation/evaluation partitions. It is reasonable to think that we would be better off using all 70000 images to train a neural network. However, we need some method for *measuring* how well a model is performing. That is the purpose of the validation set—to measure performance.

If we measure performance directly on the training dataset, we might trick ourselves into thinking that the neural network will perform very well when it is eventually deployed as part of an application, when in reality the network might only perform well specifically on the examples found in the training dataset.

Similarly, the evaluation partition is only used to compare performance after hyper-parameter tuning.

## Loading MNIST Using PyTorch

We've discussed notation and general concepts, but how would we write this out in code? Here is an example of how to load the MNIST dataset using PyTorch.

```python
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Normalize, ToTensor

# Location in which to store downloaded data
data_dir = "../Data"

mnist_xforms = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

# Load data files (training and validation partitions)
train_data = MNIST(root=data_dir, train=True, download=True, transform=mnist_xforms)
valid_data = MNIST(root=data_dir, train=False, download=True, transform=mnist_xforms)

# Data loaders provide an easy interface for interacting with data
train_loader = DataLoader(train_data, batch_size=len(train_data))
valid_loader = DataLoader(valid_data, batch_size=len(valid_data))

# Force the train loader to give us all inputs and targets
X_train, y_train = next(iter(train_loader))
X_valid, y_valid = next(iter(valid_loader))

print("Training input shape    :", X_train.shape)
print("Training target shape   :", y_train.shape)
print("Validation input shape  :", X_valid.shape)
print("Validation target shape :", y_valid.shape)
```

<details class="question">
<summary><strong>Question:</strong> What do you expect to see as this program's output?</summary>
<div class="answer">
<strong>Answer:</strong>
<pre>
Training input shape    : torch.Size([60000, 1, 28, 28])
Training target shape   : torch.Size([60000])
Validation input shape  : torch.Size([10000, 1, 28, 28])
Validation target shape : torch.Size([10000])
</pre>
This is slightly different than what we discussed. PyTorch expects us to use this dataset with a convolutional neural network.
</div>
</details>

## Similarity Digit Classifier

Before we get into training NNs, we will start with a non-ML classifier. This will provide a nice comparison, and show that ML must be *learning* something beyond simple comparisons.

Let's try to solve the following problem:

<details class="question">
<summary><strong>Question:</strong> Given the MNIST dataset and also an image of an unknown digit, how would you decide which digit is represented in the unknown image?</summary>
<div class="answer">
<strong>Answer:</strong> One method would be to find an "average" image for the ten separate digits, and then compare the unknown image to the ten averages and assign the unknown label as that of the closest average image.
</div>
</details>

For reference, here is what the "average" looks like for each of the ten digits.

![Average of the ten MNIST digits from the training dataset.](/img/MNISTAverages.png)

Before we show a solution, however, we should take a guess at how well a random guesser might perform.

<details class="question">
<summary><strong>Question:</strong> What percent of the time would you be correct in guessing digits if you were guessing at random?</summary>
<div class="answer">
<strong>Answer:</strong> If you are equally likely to guess any of the ten digits, then you would be right around 10% of the time ($\frac{1}{10}$).
</div>
</details>

And now some code for finding the most similar digit.

```python
from math import inf
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Normalize, ToTensor

data_dir = "../Data"
mnist_xforms = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

train_data = MNIST(root=data_dir, train=True, download=True, transform=mnist_xforms)
valid_data = MNIST(root=data_dir, train=False, download=True, transform=mnist_xforms)

train_loader = DataLoader(train_data, batch_size=len(train_data))
valid_loader = DataLoader(valid_data, batch_size=len(valid_data))

X_train, y_train = next(iter(train_loader))
X_valid, y_valid = next(iter(valid_loader))

# Get the average for each digit based on all training examples
digit_averages = {}
for digit in range(10):
    digit_averages[digit] = X_train[y_train == digit].mean(dim=0).squeeze()

def get_most_similar(image: torch.Tensor, averages: dict):
    closest_label = None
    closest_distance = inf
    for label in averages:
        distance = (image - averages[label]).abs().mean()
        if distance < closest_distance:
            closest_label = label
            closest_distance = distance
    return closest_label

num_correct = 0
for image, label in zip(X_valid, y_valid):
    num_correct += label == get_most_similar(image, digit_averages)

print(f"Percent guessed correctly: {num_correct/len(X_valid)*100:.2f}%")
```

<details class="question">
<summary><strong>Question:</strong> Take a guess at the accuracy of our similarity-based model.</summary>
<div class="answer">
<strong>Answer:</strong> This model is correct about 66.85% of the time.
</div>
</details>
