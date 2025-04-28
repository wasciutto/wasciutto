My training framework uses the concept of "Trainers" to define classes that create parameters for questions. In this case, I have `AdditionTrainer` classes that output two parameters, which are the first and second operand of the addition problem. To create completely random questions, I always have a randomized trainer to provide a foundation for more sophisticated trainers. For addition, that class signature looks like this:

<div style="margin-top: 4rem;"></div>

![sample data](/assets/2025-04-14/param_generator.png)

<div style="margin-top: 4rem;"></div>

The idea here is that once I've developed a strategy to provide more tailored questions to the user, I can create a new trainer to implement that strategy. And it also enables me to have multiple stragies and see how they compare to one another!