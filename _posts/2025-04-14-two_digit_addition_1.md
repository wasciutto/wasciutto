---
layout: post
title: Two Digit Addition Trainer
---

I've had the idea for a while to build a math trainer application with the aim of producing questions that the user would get right, on average, about 80% of the time. Further, I wanted to be able to generate questions within defined ranges - no hand-creating questions!

To keep things simple, the first iteration of my application would produce 2 digit addition problems with two operands. For example, `31 + 45`, or `95 + 01`.

This might seem simple to predict - why not just scale the numbers up and down according to how many questions the user is getting right? Larger numbers are generally harder to compute. But I wanted a more granular capability - `50 + 50`, for example, is using two operands that are bigger than `18 + 37`, but nobody would argue the latter is more difficult. 

Sure, I could hand-create rules, like "numbers with 0s in the first digit are easier," but I wanted to develop a way to create randomized problems for a wide variety of operations without having to hand-build each one. What makes a set of operands more difficult to subtract is going to be different from, say, multiplication - and this is before even getting near multi-operand problems!

### Basic Application & Data Setup

I won't go into the design of the actual quizzer application too much here; instead I want to focus on the data. In short, I created a simple framework that is both flexible enough to work as a CLI application, and also serve a REST API that could eventually drive a visual, web-based frontend. The current CLI is very minimal; just enough to get the job done:

<div style="margin-top: 4rem;"></div>


<video width="400" controls autoplay>
    <source src="/assets/2025-04-14/cli_demo.mp4" type="video/mp4">
</video>


<div style="margin-top: 4rem;"></div>


2-digit addition problems are simple enough that, given enough time, users with enough basic arithmetic experience could acheive close to 100% accuracy. So, a time limit of 5 seconds (ajustable as configuration) was enforced to ensure that the user would get enough questions wrong. I chose to, for now, not cut off the question when the time runs out - the full time to complete a question could be useful data! Instead, I silently marked the question incorrect if it wasn't answered within the time limit.

The data for collected questions is structured like this:

<div style="margin-top: 4rem;"></div>

![sample data](/assets/2025-04-14/sample_data.png)

<div style="margin-top: 4rem;"></div>

My training framework uses the concept of "Trainers" to define classes that create parameters for questions. In this case, I have `AdditionTrainer` classes that output two parameters, which are the first and second operand of the addition problem. To create completely random questions, I always provide a randomized trainer to serve as a foundation for more sophisticated trainers.

The next thing I needed to do was collect some data so that I could attempt to make inferences based on my own response behavior. 


### Gathering Data

Using the randomized trainer, I completed the somewhat grueling (but educational) process of grinding out hundreds of two-digit addition questions.

How many should I answer? Here we get into the idea of "parameter space" - how many possible two-digit addition questions are there? Including zero, that's 100 x 100 for 10,000 possible addition questions. So, even in a relatively simple scenario, the number of possible questions is high enough that I can't explore the entire space in a reasonable amount of time (which is good, or this wouldn't be interesting!).

I decided to gather 360 answers - this was based more off of my own endurance and what I figured to be a rough representative sample for 10,000 data points than anything statistically premeditated. Turns out (in retrospect) that gets us pretty close to 95% confidence, which sounds great to me for a demo!

Here's the result of my training:

<div style="margin-top: 4rem;"></div>

![sample data](/assets/2025-04-14/random_results.png)

<div style="margin-top: 4rem;"></div>

The results lined up well with intuition. The biggest cluster of incorrect questions occurs where there are two large operands. Answers at the low end of either axis are likely to be correct (e.g. `0 + 99` is easy). What kind of inferences could I make from this data?

### Developing a Modeling Strategy

First, I had to address the question of how exactly to generate questions that the "user will get correct 80% of the time." How could this be broken down into a data problem?

My (naive) initial thought was to give the "correct" / "incorrect" (`1` or `0`) as my input (`X`) value, and have it produces two `Y` values: `operand_1` and `operand_2`. I could then give a fractional input value of `.8` as an input to represent an 80% chance of the question being correct, and get two operands that fit to this number as an output. 

While this lined up with the verbal expression of the problem, just a little though showed me this approach is quite backwards - all of the predictive information is on the side of the operands - what pattern could possibly be inferred from "correct" / "incorrect" alone that could expand out to two operands? Also, there's no room for variation here - how would I generate a *unique* sequence of operands that all have that same 80% chance of the user getting correct?

My next idea was a little different: keep riffing on the random operand generation. Instead of creating operands that attempt to hit some target value, I can randomly generate operands, *then* score those. My output would be some value between 0 and 1: the modeled chance a user would get a question with those two operands correct. 

For example: `operand_1` is randomly assigned `14`, and `operand_2` is `37`. Those two operands go into a model, and out comes, say, `.67`. This predicts a 67% chance the user gets the question `14 + 37 = ?` correct within 5 seconds: too difficult! What I could then do is just keep generating parameters and scoring them until I get a result of `.8`. Well, not exactly - that wasn't very precise in itself! Instead, a range - such as `.75` to `.85` made a good approximation for my purposes.

### Time to Model!

Now that I had a plan, it was time to try it out with a model. Something I learned from years of working with data scientists is to always start with a linear model, then try and beat that.


Because I intended to eventually try to apply a deep learning model to this problem, I implemented the linear model with Keras. For those interested, here's the setup I used to achieve a linear model, using the `mean_squared_error` loss function:

<div style="margin-top: 4rem;"></div>

```python
model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(2,)),
        normalizer,
        tf.keras.layers.Dense(1, activation='relu')
])

model.compile(
    optimizer='adam',
    loss='mean_squared_error',
    metrics=['accuracy']
)

history = model.fit(
    train_features, train_label,
    epochs=50,
    batch_size=4,
    validation_data=(test_features, test_label),
    verbose=1
)
```

<div style="margin-top: 4rem;"></div>

This model resulted in an accuracy of about 70% when run against the test data. Unfortunately, I didn't preserve sample outputs from this data. However, I do I have fun visualization to show what pattern the model found:

<div style="margin-top: 4rem;"></div>

![linear predictions](/assets/2025-04-14/linear_predictions.png)

<div style="margin-top: 4rem;"></div>

What we are looking at here is the original random data used to train the linear model, superimposed on a *heatmap* of the model's predictions at every point in the space of all possible `operand_1` and `operand_2` combinations.

Where this background heatmap is more blue, the user (me) is predicted to be more likely to get the question wrong. For example, the top right corner of the chart represents a prediction in the neighborhood of me having a 20% chance of getting that question correct. Conversely, the very green bottom left of the chart predicts that I will almost certainly get it correct.

The red line represents my "sweet spot" - the exact point where the model predicts the user will have an 80% chance of getting the question correct. The `LinearTrainer` would thus produce a cloud of randomized questions, all falling within +/-5% of this red line.

This is an interesting scenario - while the accuracy of 70% does a decent job of generating questions that meet my criteria, it reveals a criteria of mine that I had internalized, but did not define: that the selected questions also *represent as broad a portion of the parameter space as possible.*

While this model technically does the job, I am only able to get operand combinations that fall near the red line, like `41 + 21` or `32 + 33`. High value operand combinations like `80 + 90` would be ignored completely. Even worse, the model overstates the difficulty of combinations like `60 + 0` or `1 + 55`, marking them as at the 80% line when they are almost certain to be an accuracy of 95%+.

Could a non-linear model do a better job?

### Deep Learning Model

Next up, I wanted to try a small neural net to see if I could better capture the non-linear elements of the data.

I decided to use binary cross-entropy as my loss function, which is supposed to work well with binary classification problems like this one. A lot of experimentation showed me that adding more layers or units did not achieve too much for my problem - probably due to the low amount of data.

For those interested, here again is the setup:

<div style="margin-top: 4rem;"></div>

```python
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(2,)),
    normalizer,
    tf.keras.layers.Dense(units=8, activation="relu"),
    tf.keras.layers.Dense(units=1, activation="sigmoid")
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=keras.losses.BinaryCrossentropy(),
    metrics=[keras.metrics.BinaryAccuracy()]
)

history = model.fit(
    train_features, train_label,
    epochs=50,
    batch_size=4,
    validation_data=(test_features, test_label),
    verbose=1
)
```

<div style="margin-top: 4rem;"></div>

And here are the results, using the same heat map view as the linear model:

<div style="margin-top: 4rem;"></div>

![binary predicitons](/assets/2025-04-14/binary_predictions.png)

<div style="margin-top: 4rem;"></div>

The gradations between correct/incorrect are harder to see here, but already it is apparent that a more organic pattern is being captured with the neural net.

To help clear make the patterns in the heat map more visible, I customized the heatmap plot colors. In this new plot, I have made values close to the 80% threshold white, so that this boundary (similar to the 80% red line in the linear plot) is easy to see:

<div style="margin-top: 4rem;"></div>

![binary predicitons fancy](/assets/2025-04-14/binary_predictions_fancy.png)

<div style="margin-top: 4rem;"></div>

Contrasted to the linear model, we can see here that the NN was able to better handle the cases along the axis: `0 + 100` and `100 + 0` are properly categorized as a 90% + chance of being correct, for example, where the linear model marked these as more difficult.

In the following graphic, we can see what problems were genereated by the `NeuralNetTrainer` and my answers to them:

<div style="margin-top: 4rem;"></div>

![binary predictions generated](/assets/2025-04-14/binary_predictions_generated_fancy.png)

<div style="margin-top: 4rem;"></div>

This is, like the linear model, technically working as intended! I'm getting roughly 80% of the questions correct.

However, there are still some problems. First, the NN model only manages to deliver the same 70% accuracy as the linear model! So while it is giving slightly more diverse questions by following a non-linear contour, it doesn't actually manage to classify more accurately. Why this is, I am not sure; perhaps linear model's overestimation of difficulty near low numbers is made up for by the difficulty of other points being underestimated.

Second, this model still delivers a very unsatisfying distribution of operand combinations. It's clear from the sampled model output above that we're only ever going to get problems that use operand combinations near that white 80% region. The entire rest of the parameter space is ignored! Surely there are a multitude of combinations in those deep green and blue portions of the space that are, in reality, near the 80% mark.

### Improving Features