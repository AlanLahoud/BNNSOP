This repository provides implementations based on the article "On the use of Bayesian Neural Networks for Data-driven Optimization Problems".


*A. What does this project do?* This project uses Machine Learning and Stochastic Mathematical Programming techniques to solve two versions of the Newsvendor Problem. The first is a classical version, and the second is a Quadratic Programming version with multiple items.


*B. How to run?* There are two main files to run. "classic_newsvendor.py" for the Classical Newsvendor experiment and "constrained_newsvendor.py" for the Quadratic Programming version of the Newsvendor experiment. These two files requires arguments from command line and we explain it below:

    *1. classic_newsvendor.py*

    1.1 Model/method
    Possible values: "ann", "bnn", "gp"
    Note that "ann" and "gp" are baselines

    1.2 Method of learning
    Possible values: "decoupled", "combined"
    if "gp" was chosen, decoupled is the only option

    1.3 Noise type while creating synthetic data
    Possible values: "gaussian", "multimodal"

    1.4 Number of seeds (how many times it runs to provide results average)
    Possible values: Any integer >= 1 (but not too high), try 3 to 10

    1.5 Number of output samples of the BNN while training (M_train)
    Possible values: Any integer >= 1 (but not too high). if using combined approach, 
    this value can't be high due to complexity. If using ANN or GP, this value does not matter.

Example to run:
python3 classic_newsvendor.py bnn decoupled gaussian 3 16


    *2. constrained_newsvendor.py*

    2.1 Model/method
    Possible values: "ann", "bnn", "gp"
    Note that "ann" and "gp" are baselines

    2.2 Method of learning
    Possible values: "decoupled", "combined"
    if "gp" was chosen, decoupled is the only option

    2.3 Number of seeds (how many times it runs to provide results average)
    Possible values: Any integer >= 1 (but not too high), try 3 to 10

    2.4 Number of output samples of the BNN while training (M_train)
    Possible values: Any integer >= 1 (but not too high). if using combined approach, 
    this value can't be high due to complexity. If using ANN or GP, this value does not matter.

Example to run:
python3 constrained_newsvendor.py bnn decoupled 3 16
