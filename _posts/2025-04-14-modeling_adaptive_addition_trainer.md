---
layout: post
title: Modeling an Adapative Addition Trainer
---

About three years ago, I heard about some findings from a [Nature Communciations paper](https://www.nature.com/articles/s41467-019-12552-4) proposing that the optimal success rate for any learning task is about 85%.

This inspired me to start building a training application framework that operates on this principle, randomly generating questions that the user would get right about 85% of the time. The application would utilize a model built from previously answered questions to generate new problems that meet this threshold, continuously adapting to a user's answers. My idea was that staying within this "sweetspot" of difficulty would keep a user more engaged with the trainer.

To keep things simple, I decided to make the first iteration of my application a 2-digit addition trainer, with the goal of generating single operator addition problems that the user had an 85% chance of getting correct. For example, `31 + 45`, or `95 + 23`.

This might initially seem trivial - why not just scale the numbers up and down according to how many questions the user is getting right? Larger numbers are generally harder to compute. But I wanted a more granular capability - `50 + 50`, for example, uses operands that are larger than `18 + 37`, but nobody would argue that the former is more difficult. 

Sure, I could hand-create rules to account for such scenarios, like "numbers with `0` in the ones column are easier," but I wanted to develop a flexible, scalable method of generating randomized problems for a wide variety of operations without having to hand-build each one. Instead of defining exactly which kind of operands make a subtraction or multiplication problem more or less difficult, I wanted to develop a modeling framework to figure that out for me.

### Basic Application & Data Setup

To keep the focus on the data & modeling, I won't get too much into the design of the actual training application here (perhaps a future post). In short, I created a simple framework that is both flexible enough to work as a CLI application, and also serve a REST API that could eventually drive a visual, web-based frontend. The current CLI is very minimal; just enough to get the job done:

<div style="margin-top: 4rem;"></div>


<video width="400" controls autoplay>
    <source src="/assets/2025-04-14/cli_demo.mp4" type="video/mp4">
</video>


<div style="margin-top: 4rem;"></div>


A glaring initial problem is that 2-digit addition problems are simple enough that, given enough time, users with basic arithmetic experience can acheive close to a 100% success rate. To account for this, I added a time constraint of 5 seconds (configurable) to ensure that the user would get enough questions wrong, thus making the goal of 85% success rate at all possible. I chose to, for the time being, not cut off the question when the time runs out - the full time to complete a question could be useful data! Instead, I silently marked the question as "incorrect" if it wasn't answered within the time limit.

I structured the collected data for answered questions like this:

<div style="margin-top: 4rem;"></div>

![sample data](/assets/2025-04-14/sample_data.png)

<div style="margin-top: 4rem;"></div>

(there's a few extra fields captured too, like timestamps)

This training framework stores the "parameters" used to populate each question that was fed to the user. For example, the question `What is 24 + 19?` is created by combining a question template with corresponding parameters. In this case, the template is `What is {operand_1} + {operand_2}?`. By storing the parameters used to create each question alongside the correct/incorrect feedback, I should have what I need to attempt to predict an answer outcome from given parameters.

The framework uses classes I named "Trainers" that define how parameters are generated for a particular type of problem template. In this case, I setup an `AdditionTrainer` class with two parameters, which are the first and second operand of the addition problem. Using this as a base, I then created a `RandomAdditionTrainer` class that generates random parameters for the addition problem (in this case, the two operators) The randomized trainer serves as a crucial foundation for more sophisticated trainers.

Next, I needed to collect some data to work with!


### Gathering Data

Using the `RandomAdditionTrainer`, I completed the somewhat grueling (but educational) process of grinding out hundreds of two-digit addition questions.

How many should I answer? Here we get into the idea of this problem's *parameter space*, which is the answer to the question: what are all of the possible (positive) two-digit addition problems? Including zero, that's 100 x 100 = 10,000 possible addition questions. As you can imagine, that's too many combinations for me to cover in a reasonable amount of time, even for this relatively simple problem. (which is good, or this wouldn't be interesting!).

I decided to gather 360 answers - admittedly, this was based more off of my own stamina than anything statistically premeditated. It turns out (in retrospect) that this gets us pretty close to 95% confidence for representing the population size of 10,000 possible problems, which sounds great to me for a demo!

Here's the result of my training:

<div style="margin-top: 4rem;"></div>

![sample data](/assets/2025-04-14/random_results.png)

<div style="margin-top: 4rem;"></div>

The scatter plot points should be read as addition problems - the point near x = 0, y = 39, for example, should be read as `0 + 39`. The feedback (correct / incorrect) is given by the point's color.

The results lined up well with my intuition. The biggest cluster of low success rates occured where there are two large operands. Answers at the low end of either axis have high success rates (e.g. `0 + 99` is easy). However, there are also plenty of the expected exceptions to these patterns. What kind of inferences could I make from this data?

### Developing a Modeling Strategy

First, I had to address the question of how exactly to generate questions that the user would get correct 85% of the time (actually, I misremembered the exact value and used an 80% threshold for the following experiment instead - but close enough). How could I break this down into a solution?

My (naive) initial thought was to give the "correct" / "incorrect" value (`1` or `0`) as my input (`X`) value, and have it produce two `Y` values: `operand_1` and `operand_2`. I could then give a fractional input value of `.8` to represent an 80% success rate, and get two operands that fit to this number as an output. 

While this lined up with the verbal expression of the problem, just a little thought showed me this approach is quite backwards - all of the predictive information is on the side of the operands - what pattern could possibly be inferred from "correct" / "incorrect" alone that could expand out to two operands? Also, there's no room for variation here - how would I generate a *unique* sequence of operands that all have that same 80% success rate?

My next idea was a little different: keep riffing on the random operand generation. Instead of creating operands that attempt to hit some target value, I can randomly generate operands, *then* classify those parameters. My output would be some value between 0 and 1: the modeled chance a user would get a question with those two operands correct. 

For example: `operand_1` is randomly assigned `14`, and `operand_2` is `37`. Those two operands go into a classification model, and out comes, say, `.67`. This predicts a 67% chance the user gets the question `14 + 37 = ?` correct within 5 seconds: too difficult! What I could then do is keep generating parameters and scoring them until I get a result of `.8` (80% success rate). Well, not exactly - that wasn't very precise in itself! Instead, a range - such as `.75` to `.85` made a good approximation [^1].

### Time to Model!

Now that I had a plan, it was time to try it out with a model. Something I learned from years of working with data scientists is to always start with a linear model, then try and beat that.

A goal of mine during this project was to get some practice with building deep learning models, so I implemented the linear model with Keras. I leaned heavily on Francois Chollet's *Deep Learning with Python* for the following modeling work, a book I deeply recommend! For those interested, here's the setup I used to achieve a linear model, using the `mean_squared_error` loss function:

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

This model resulted in an accuracy of about 70% when run against the test data. Not bad! Unfortunately, I didn't preserve sample outputs from this data. However, I do I have fun visualization to show what pattern the model found:

<div style="margin-top: 4rem;"></div>

![linear predictions](/assets/2025-04-14/linear_predictions.png)

<div style="margin-top: 4rem;"></div>

What we are looking at here is the original scatterplot of the random data used to train the model, superimposed on a *heatmap* of the model's predictions over the entire parameter space [^2].

Where this background heatmap is more blue, the user (me) is predicted to be more likely to get the question wrong. For example, the top right corner of the chart represents a prediction in the neighborhood of me having a 20% chance of getting that question correct. Conversely, the very green bottom left of the chart predicts that I will almost certainly get the question correct.

The red line represents my "sweet spot" - the exact point where the model predicts the user will have an 80% success rate. The `LinearTrainer` would thus produce a cloud of randomized questions, all falling within +/-5% of this red line.

This is an interesting scenario - while the model's accuracy of 70% does a decent job of generating questions that meet my criteria, it reveals a criteria of mine that I had internalized, but had not defined: that the selected questions *represent as broad a portion of the parameter space as possible.*

While this model technically produces the correct difficulty level, I am only able to get operand combinations that fall near the red line, like `41 + 21` or `32 + 33`. High value operand combinations like `80 + 90` would be ignored completely. Even worse, the model overstates the difficulty of combinations like `60 + 0` or `1 + 55`, marking them an 80% success rate, when they are almost certain to be at a 95%+.

Could a non-linear model do a better job?

### Neural Net Model

Next up, I wanted to try a small neural net to see if I could better capture the non-linear elements of the data.

I decided to use binary cross-entropy as my loss function, which is supposed to work well with binary classification problems like this one. A lot of experimentation showed me that adding more layers or units did not achieve too much for my problem - probably due to the low amount of data. As a result, I used a simple two layer approach: the first with 8 units, and the second to consolidate them into a single output.

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

To help clear make the patterns in the heat map more visible, I customized the heatmap plot colors. In this new plot, I have made values close to the 80% success rate threshold white, so that this boundary (similar to the 80% red line in the linear plot) is easy to see:

<div style="margin-top: 4rem;"></div>

![binary predicitons fancy](/assets/2025-04-14/binary_predictions_fancy.png)

<div style="margin-top: 4rem;"></div>

Contrasted to the linear model, we can see here that the NN was able to better handle the cases along the axis: `0 + 100` and `100 + 0` are properly categorized as a 90% + success rate, for example, where the linear model marked these as more difficult.

In the following graphic, we can see what problems were genereated by the `NeuralNetTrainer` and my answers to them:

<div style="margin-top: 4rem;"></div>

![binary predictions generated](/assets/2025-04-14/binary_predictions_generated_fancy.png)

<div style="margin-top: 4rem;"></div>

This is, like the linear model, technically working as intended! I'm getting roughly 80% of the questions correct when testing the application [^3].

However, there are still some problems. First, the NN model only manages to deliver the same 70% accuracy as the linear model! So while it is giving slightly more diverse questions by following a non-linear contour, it doesn't actually manage to classify more accurately. Why this is, I am not sure; perhaps the linear model's overestimation of difficulty near low numbers is made up for by the difficulty of other points being underestimated.

Second, this model still delivers a very unsatisfying distribution of operand combinations. It's clear from the sampled model output above that we're only ever going to get problems that use operand combinations near that white 80% region. The entire rest of the parameter space is ignored! Surely there are a multitude of combinations in those deep green and blue portions of the space that are, in reality, near the 80% success rate.

### Feature Engineering!

Eventually I started to realize that my formulation of the problem might be holding my modeling back. I had been feeding the model *numbers*, but was that really the right modality? When we add, we operate at the level of digits. Adding `89 + 12`, we're all grabbing that `9` and `2` and summing them, carrying over the `1`. What makes most 2-digit problems any challenge to solve in 5 seconds is really the *carry*. It's why `68 + 37` is harder than `68 + 31`; the latter has no carry. And what makes certain problems easier despite their size, like `75 + 35` or `37 + 33`, is that their ones-columns "click" together into 10.

As mentioned earlier, I have no intention of handling these cases manually; I want the model to pick up these patterns on its own. The only way to do that is to make the model *digit aware*. Instead of feeding the model numbers, I should give it the digits!

Thus, instead of `operand_1` and `operand_2`, I re-engineered those features into `operand_1_digit_1`, `operand_1_digit_2`, `operand_2_digit_1`, and `operand_2_digit_2`. `35` and `75` becomes `3`, `5`, `7`, and `5` [^4].

Instead of re-architecting the stored data into digits, I simply added a transformation function that converts the numbers into digits right before they are fed into the model. A fun result of this is I can still use the same heatmap to plot this new, 4-parameter input space in two dimensions. Starting again with a linear model:

<div style="margin-top: 4rem;"></div>

![digits linear](/assets/2025-04-14/digits_linear.png)

<div style="margin-top: 4rem;"></div>

As someone newer to modeling, this was a *fascinating* visualization! I now had a grid of grids, each square serving as a mini-model for every increment of 10. With the expanded features, the model now picks up on the pattern that higher ones-column digits make the problem harder, independently of the tens-column digits. For example, `29 + 19` is properly predicted as difficult, while `25 + 25` is solidly easy.

Now, with predictions again falling along the white regions of this space, the variation of outputted questions is much higher!

The model's accuracy is also respectably higher: about 75%, a solid improvement.

Finally, what about combining the digit-based parameters with a neural net?

<div style="margin-top: 4rem;"></div>

![digits nn](/assets/2025-04-14/digits_nn.png)

<div style="margin-top: 4rem;"></div>

While I got the same organic curve in each of these "mini-plots" that probably gives more varied outputs than the linear, again I'm failing to beat the linear model, tying it with 75% accuracy. Is this some sort of convergence due to the low amount of data, or perhaps the sheer simplicity of the problem? I'm not sure, but it's something I'll look out for in future modeling.

There are clearly improvements to seek here - sticking out to me is that sparse cluster of green point in the top right, which did not manage to get that region categorized as a higher success rate. 


### Ideas for Model Improvement

Getting more data is always a solution (it's quite emphatically the first thing Chollet suggests), and would probably help my modeling accuracy more than anything; however, this case represents a very minimal parameter space. These problems will only get more complex, and how efficient would it be to gather more data than the 360 / 10,000 ratio I've gotten here?

During an earlier iteration of the project, I experimented with training the model, answering questions from the new model, then re-training on that. While I moved on to training from purely random data for experimental consistentcy, there is probably value in re-training from modeled data to reduce the parameter space and gather more information around relevant spaces.

The most useful dimension to add that I have immediate access to is response time. This would give the model another dimension of difficulty to classify with instead of just the digits. This could compensate for easy questions I got wrong due to hasty answering, and give more difficulty to questions I got right just barely in time.

Another potentially useful dimension is the question's time or position within particular sessions and sets. I didn't get too much into it in this post, but questions are delivered in "sets" of 10 (adjustable) and within the context of a sitting - a session. As a user is likely to get fatigued as the session goes on, it is probably useful to track what position a question is within a set or session, either by time or an ordinal position (e.g. question #1, #2, etc.). This is a practice I saw used by default in fMRI studies during my time in UNC's psychology labs, so I know it is an imporant feature to eventually control for.

Finally, the effect of a user learning this task over time should probably be taken into account; perhaps by using each record's general timestamp. Questions that were marked wrong a relatively long time ago in the answer set would certainly be "stale," not reflecting the user's more practiced arithmetic skills. I would love to eventually get into some timeseries-based modeling, where more recently answered questions are weighted much more heavily than older ones; perhaps experimenting with RNNs or LTSMs.


### Future Plans

- Expanding to handle more than two operands / one operator
- Supporting other operators like `-`, `*`, and `/`, and combinations of both
- Updating the model dynamically as the user takes the quiz; every 10 questions, or even after every question
- Tracking data from multiple users
- Non-math trainers - I have a prototype in the works for musical ear training using the same framework

<div style="margin-top: 4rem;"></div>

Thanks for reading! 

Feel free to share with me any suggestions, comments, feedback, etc. you may have. I'm still new to modeling, so I would love to know if I'm using any terms inaccurately, am getting some foundational assumptions incorrect, am using models/modeling parameters incorrectly, etc.

You can reach me via email at wjasciutto@gmail.com, or on [LinkedIn](https://www.linkedin.com/in/will-asciutto-973879118/).


<div style="margin-top: 4rem;"></div>

### Footnotes

[^1]: I realized that this solution might be untenable for problems with a larger parameter space, taking too much compute time to hit a score within my defined range. To account for this, I made sure to isolate the parameter generation strategy from the Trainer. This would allow me to, in the future, create a more efficient method for generating parameters to score than random - perhaps through an RL model. For these purposes, I found random generation to be more than adequate.
[^2]: The method I used here was to leverage `np.meshgrid` to create a matrix of all possible combinations of operands, then feed those into the model. Matplotlib's `imshow` could then render the output matrix as a background image for the grid.
[^3]: While this output data matches the gradient of the model properly, I'm not sure why it is a little under the white region here: either the generated data is unshooting the range I wanted by a bit, or the color grading is a little off. Will investigate as I move forward.
[^4]: I put in a good amount of consideration as to whether I should try and still "group" single digits by the number they compose. E.g, grouping together `3` and `5` here. I decided, based on other examples from the Chollet guide, that this wasn't necessary - the model can probably make that inference from order alone. But this is something I need to learn more about.