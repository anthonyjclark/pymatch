# 1. Introduction

> Any sufficiently advanced technology is indistinguishable from magic.
>
> — Arthur C. Clarke

**Goal:** Provide a concise walk-through of all fundamental neural network (including modern deep learning) techniques.

I will not discuss every possible analogy, angle, or topic here. Instead, I will provide links to external resources so that you can choose which topics you want to investigate more closely. I will provide minimal code examples when appropriate.

Useful prior knowledge:

- Matrix calculus
- Programming skills
- Familiarity with computing tools

## Background

Artificial neurons date back to the 1940s and neural networks to the 1980s. These are not new techniques, but we surprisingly still have a lot to learn about how they work and how to best create them. Research into neural networks accelerated in the late 2000s as we found ourselves with more data and more compute power.

Neural networks (NN) are a type of machine learning (ML) technique. ML falls under the artificial intelligence (AI) umbrella. AI is a broad area, and it doesn't hurt to think of it as including any system that appears to do something *useful* or *complex*.

You'll find that techniques often start out as AI, but then we remove that label after we start to better understand them. ML is just one type of AI, and it comprises all techniques that automatically learn from data. NNs learn from data, and specifically they do so using very little input from the designer.

![NNs are a subset of ML, which is a subset of AI.](/img/AI.svg)

All of this is a bit vague, so let's discuss some specific applications. Maybe we want a NN to:

- Tell us if an Amazon review is positive or negative based on text alone.
- Tell us if an image contains a cat or a dog.
- Translate an English sentence to German.
- Tell us where in an image we can find a boat.
- Automatically generate a caption for an image.
- Direct a robot around a building.
- Play a board game or a video game.
- Tell us about the orientation of a person's limbs for a virtual reality game.
- Prevent an autonomous car from driving off the road.
- Group together all users of a social network that are likely to listen to the same music.
- Create a new piece of art.
- Predict the sale price of a house.
- Predict the future sale price of an investment.
- Suggest products to purchase or movies to watch.
- Diagnose an injury from an X-ray CT scan.
- Automatically summarize a news article.
- Label a news article as fake or real.

This is just a small subset of what we could do. Nearly all applications have the same basic flow:

![General flow of data in a NN.](/img/MLProgram.svg)

The core of the NN is the ability to take an input, perform some mathematical computations, and then produce the output. The "learning" part includes comparing the output to a known-to-be-correct output (aka the "label" or "target") and then using this comparison to iteratively improve the NN.

This setup, where we know the correct output, is known as "supervised learning." Later parts of this guide will touch on "unsupervised learning" and "reinforcement learning," but it is safe to say that most ML applications are in the area of supervised learning.

<details class="question">
<summary><strong>Question:</strong> What might be the input, output, label, and criterion if we want an NN to distinguish between pictures of cats and pictures of dogs?</summary>
<div class="answer">
<strong>Answer:</strong> The input would be an image, the output would be a guess of cat or dog, the label would be the actual contents of the image, and the criterion should have something to do with if the output guess was correct or not.
</div>
</details>

## Additional Material

- [A Whirlwind Tour of Python (free Book)](https://github.com/jakevdp/WhirlwindTourOfPython "A Whirlwind Tour of Python")
- [The Matrix Calculus You Need For Deep Learning (web-page)](https://explained.ai/matrix-calculus/)
- [Introduction — Dive into Deep Learning (free Book)](https://d2l.ai/chapter_introduction/index.html "Introduction — Dive into Deep Learning")
