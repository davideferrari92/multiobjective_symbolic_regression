# Multi-objectie Symbolic Regression

**Symbolic Regression** is a mathematical technique based on genetic programming that search in the space of mathematical equations to find one or more that best fit a given learning task bsaed on a dataset.
In this repository you will find an implementation from scratch in Python that allow the versatile implementation of any fitness function (loss function) and use any number of them during one training task. This is called "*multi-objective training*". It consists of evaluating more than one fitness function per training loop and consider all of them in a multidimensional space in which a datapoint represent one of the generated models, and its coordinates represent the performance value for every single fitness measure.

## The genetic approach
Genetic algorithms are inspired by how the biological DNA evolution occurs in nature. Here are some key definitions:

```Population``` It is a finite set of entities that represent, in this case, mathematical equations. 

```Individual``` Is one of the independent entities that constitutes a population. In the Symbolic Regression case, they are mathematical expressions. Each of them is constituted by an arbitrary sequence of ```Operations``` (e.g., sum, subtractions, multiplications, etc.), ```Constants``` (e.g., 1 or 2), and also variable ```Features``` taken from a dataset that come from real-world measurements (e.g., weight and height of a patient in a hospital).

A basic representation of an individual is reported as a binary tree in the figure below.

<p align="center">
<img src="/doc/figure_expression.png" alt="Simple Expression" style="width:40%; border:0;">
</p>

Internal nodes are given by operations (```Operation Nodes```, or ```OpNode```), while the leaves are terminal nodes with variables or constants (```Feature Nodes``` or ```FeatNode```).
Operations are chosen from a predefined set based on what possibilities the experiments aim to explore; we allow:
- *addition*
- *subtraction*
- *multiplication*
- *division*
- *natural logarithm*
- *exponential*
- *power*
- *square root*
- *maximum*
- *minimum* 

The genetic approach consist of a random creation of ```N``` individuals belonging to a population ```P``` producing a sequence of independent models each with their own structure, operations, constants, and features. Randomness grants a great variability in the nature and behaviour of the individual models. 

The next step is a loop of ```G``` generations in each of which the algorithm create another set of ```N``` individuals from the original population and, with a tournament selection, applying to them a set of genetic operations trying to produce a modified and better version. The generation of these offsprings doubles the size of ```P``` to ```2N``` elements, so that at the end of each generation the Non Dominanst Sorting Algorithm described below re-organize the population to discard the ```N``` worst ones and keep only the ```N``` best. This is the essence of the propagation of the best genes from one generation to the following.

This process is depicted in the figure below.

<p align="center">
<img src="/doc/genetic_training.png" alt="Genetic Training" style="width:70%; border:0;">
</p>

In this repository, the **Genetic Operations** we implemented are:

- *point mutation* (an ```OpNode``` and its sub-tree are replaced by a newly randomly generated sub-tree)
- *crossover* (an ```OpNode``` and its sub-tree is replaced by an ```OpNode``` and its sub-tree from another individual of the pool)
- *node insertion* (an new ```OpNode``` is inserted at a randomly selected point in the tree making it deeper of one level)
- *node deletion* (a random ```OpNode``` is deleted and its operators are shifted one level above)
- *leaf mutation* (a random ```FeatNode``` is replaced by another from the allowed ones)
- *operator mutation* (a random ```OpNode``` operation is replaced by another with the same number of operators, i.e., arity)
- *simplification* (the individual can be a non-minimal representation of an expression and therefore, although being numerically equivalent, a reorganized and simplified tree is considered in place of the selected one; e.g., $a+a+a$ becomes $3*a$) 
- *recalibration* (to avoid getting stuck in local minima in the constants optimization process, with recalibration we randomly reset the constants before optimizing them so to better explore the space of possible solutons) 

## Multi-objective training using Non Dominant Sorting Algorithm (NSGA-II)
The Non Dominant Sorting Algorithm allow to sort individuals based on the performance measures in a multi-dimensional space typical of a multi-objective training.

The concept of dominance is used to identify all the individuals whose performance is not worse than those of any other one. There may be more than one individual that satisfy this requirement and they are said to belong to the "First Pareto Front", the set of equally most optimal individual. This is depicted in the figure below as $R_1$. The remaining individuals are recursively attributed to the following pareto fronts $R_2, R_3, ..., R_n$ until the whole population is assigned.

Here is a pictorial representation of a 2-dimensional pareto front from ([Mergos and Sextos, 2018](https://www.researchgate.net/publication/329870692_Multi-objective_optimum_selection_of_ground_motion_records_with_genetic_algorithms))

<p align="center">
<img src="/doc/Pareto-optimal-solutions.png" alt="Pareto Front" style="width:70%; border:0;">
</p>

Within each pareto front, the individuals are sorted to maximize the differences between individuals, so those who are further to each other in the fitness function multi-dimensional space are set to be higher in the ranking.

Once sorted the entire population of ```2N``` elements (original population + the offsprings), we simply discard the less performing ```N``` elements, propagating to the next generation only the bst ```N```.
A well perfroming training process sorts some of the new offsprings in the higher part of the ranking to report an improvement of the predictive performance generation after generation.

At the end of training, the first pareto front contains the individuals that are not worse that any one else in the population and are therefore equally optimal for the task.
We can identify the most balanced individual as the one closest to the origin of the fitness functions space.

The algorithm is depicted in the figure below.

<p align="center">
<img src="/doc/NSGA_Dominance.png" alt="NSGA-II Algorithm" style="width:70%; border:0;">
</p>

# Multi-objective Symbolic Regression for Score Systems Management


## Example for using this library
You can find an iPython notebook in the examples folder that shows how this Python library can be used.

# Known Issues
- This implementation has been developed to work using ```sympy==1.9```. We are aware of execution problems when using with newer versions.

# Contacts
Feel free to get in touch writing to [davide.ferrari@kcl.ac.uk](mailto:davide.ferrari@kcl.ac.uk)

# Citations

## The methodology, code, and applications

**Davide Ferrari, Veronica Guidetti, Federica Mandreoli**:
*Multi-Objective Symbolic Regression for Data-Driven Scoring System Management*

2022 IEEE International Conference on Data Mining (ICDM), Orlando, FL, USA

See more on [10.1109/ICDM54844.2022.00112](https://doi.org/10.1109/ICDM54844.2022.00112)

```bibtex
@INPROCEEDINGS{ferrari2022multiobjective,
  author={Ferrari, Davide and Guidetti, Veronica and Mandreoli, Federica},
  booktitle={2022 IEEE International Conference on Data Mining (ICDM)}, 
  title={Multi-Objective Symbolic Regression for Data-Driven Scoring System Management}, 
  year={2022},
  volume={},
  number={},
  pages={945-950},
  doi={10.1109/ICDM54844.2022.00112}}
```

## A clinical benchmark
**Davide Ferrari, Veronica Guidetti, Yanzhong Wang, Vasa Curcin**:
*Multi-objective Symbolic Regression to Generate Data-driven, Non-fixed Structure and Intelligible Mortality Predictors using EHR: Binary Classification Methodology and Comparison with State-of-the-art*

Proceedings of the American Medical Informatics (AMIA) Annual Symposium, Washington DC, Nov. 2022

See more at: [https://scholar.google.it/citations?view_op=view_citation&hl=en&user=5zwLd3IAAAAJ&citation_for_view=5zwLd3IAAAAJ:W7OEmFMy1HYC](https://scholar.google.it/citations?view_op=view_citation&hl=en&user=5zwLd3IAAAAJ&citation_for_view=5zwLd3IAAAAJ:W7OEmFMy1HYC)

```bibtex
@inproceedings{Ferrari2022,
   author = {Davide Ferrari and Veronica Guidetti and Vasa Curcin and Yanzhong Wang},
   journal = {AMIA 2022, American Medical Informatics Association Annual Symposium,
Washington, DC, USA, November 5-9, 2022},
   publisher = {AMIA},
   title = {Multi-objective Symbolic Regression to Generate Data-driven, Non-fixed
Structure and Intelligible Mortality Predictors using EHR: Binary
Classification Methodology and Comparison with State-of-the-art},
   url = {https://knowledge.amia.org/76677-amia-1.4637602/f006-1.4642154/f006-1.4642155/877-1.4642417/511-1.4642414},
   year = {2022},
}
```