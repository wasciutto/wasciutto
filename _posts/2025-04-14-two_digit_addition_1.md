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


The first thing I needed to do was collect some data so that I could attempt to make inferences based on my own response behavior. 

My training framework uses the concept of "Trainers" to define classes that create parameters for questions. In this case, I have `AdditionTrainer` classes that output two parameters, which are the first and second operand of the addition problem. To create completely random questions, I always have a randomized trainer to provide a foundation for more sophisticated trainers. For addition, that class signature looks like this:

<div style="margin-top: 4rem;"></div>

![sample data](/assets/2025-04-14/param_generator.png)

<div style="margin-top: 4rem;"></div>

The idea here is that once I've developed a strategy to provide more tailored questions to the user, I can create a new trainer to implement that strategy. And it also enables me to have multiple stragies and see how they compare to one another!

### Gathering Data

So, using the randomized trainer, I completed the somewhat grueling (but educational) process of grinding out hundreds of two-digit addition questions.

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