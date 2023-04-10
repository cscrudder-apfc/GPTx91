## Project developed for the Alaska Permanent Fund Corporation by Colton Scrudder

# This project is a neural network that evaluates bond data from Aladdin. It uses four inputs and estimates the z-spread of a bond:

    # z-spread = f(duration, credit rating, risk country, sector)

# The folder, model_development, contains many iterations of the ML model, including a Bayesian optimization model to select macroparameters for the model. 

# The folder, model_use_development, contains iterations that improve the base model's functionality. Each iteration has a number at the end for the order they were developed in but not continuity between them.