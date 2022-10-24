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
- *simplification* (the individual can be a non-minimal representation of an expression and therefore, although being numerically equivalent, a reorganized and simplified tree is considered in place of the selected one; e.g., ```a+a+a``` becomes ```3*a```) 
- *recalibration* (to avoid getting stuck in local minima in the constants optimization process, with recalibration we randomly reset the constants before optimizing them so to better explore the space of possible solutons) 

## Multi-objective training using Non Dominant Sorting Algorithm (NSGA-II)
TODO

<p align="center">
<img src="/doc/NSGA_Dominance.png" alt="NSGA-II Algorithm" style="width:70%; border:0;">
</p>

## Example for using this library
You can find an iPython notebook in the examples folder that shows how this Python library can be used.

# Contacts
Feel free to get in touch writing to [davide.ferrari@kcl.ac.uk](mailto:davide.ferrari@kcl.ac.uk)

# Citations
Please use the following references to cite our work.

```
@INPROCEEDINGS {ferrari2022,
  author = {D. Ferrari and V. Guidetti and F. Mandreoli},
  title = {Multi-Objective Symbolic Regression for Data-Driven Scoring System Management},
  booktitle = {2022 IEEE International Conference on Data Mining (ICDM)},
  publisher = {IEEE Computer Society},
  address = {Orlando, FL, USA},
  keywords = {scoring systems, genetic programming, multi-objective symbolic regression},
  year = {2022},
  month = {nov}
  volume = {},
  issn = {},
  pages = {},
  doi = {},
  url = {},
}
```
