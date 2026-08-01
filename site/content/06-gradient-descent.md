# 6. Gradient Descent

Optimizing a neural network follows this process:

1. Prepare dataset(s) (e.g., training, validation, evaluation).
2. Set hyperparameters (e.g., learning rate, number of epochs).
3. Create the model.
4. Train the model.

We'll go into more details starting with training the model.

## Batch Gradient Descent

All examples thus far have used batch gradient descent (BGD). All gradient descent methods are iterative, meaning we continually make small changes to the parameters until we are satisfied or run out of time. BGD looks something like this:

```text
for each epoch
    1. compute gradient with respect to all examples
    2. average gradients across all examples
    3. update parameters using averaged gradients
```

In batch gradient descent, we compute all gradients at once and average them across all examples, resulting in the parameters being updated a single time each epoch. This has the advantage of smoothing out the effect of any outliers and leveraging parallel computation.

## Stochastic Gradient Descent

In stochastic Gradient Descent (SGD) we update parameters $N$ times per epoch—once per example.

The **stochastic** part of SGD refers to a random shuffling of the examples each epoch.

```text
for each epoch
    randomly shuffle all examples
    for each example
        1. compute gradient with respect to single example
        2. update parameters using gradient
```

Although we update the parameters more frequently, not all updates are good since outliers will make the model perform worse in the general case.

## Mini-Batch Stochastic Gradient Descent

Mini-Batch SGD provides a middle ground. We chunk the input into some number of batches and take the average gradient over each batch.

```text
for each epoch
    randomly distribute examples into batches
    for each batch
        1. compute gradient with respect to all examples in batch
        2. average gradients across all examples in batch
        3. update parameters using averaged gradients
```

This enables us to get the best of both worlds:

- less susceptible to outliers and noise,
- a good number of updates per epoch, and
- good utilization of computing resources.

<details class="question">
<summary><strong>Question:</strong> What batch size turns Mini-Batch SGD into BGD? What batch size turns Mini-Batch SGD into SGD?</summary>
<div class="answer">
<strong>Answer:</strong> $N$ and $1$, respectively.
</div>
</details>

<details class="question">
<summary><strong>Question:</strong> Will all batches be the same size?</summary>
<div class="answer">
<strong>Answer:</strong> No. The last batch is frequently smaller than all other batches. It contains the leftovers.
</div>
</details>
