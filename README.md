# HeterRecomPortfolio
This system's purpose is to serve as the framework for testing and evaluation of recommmender systems of various heterogenous architectures. It allows for testing fairness and proportionality of different types of basic recommenders aggregations. Sequential evaluation makes it possible to simulate online evaluation. 

### Code conventions:
Names of abstract classes begin with letter "A". For public class is usually created a separate file. Variables and functions are named in Camel case.


### Architecture description:
Class which represents portfolio structure definition is named [APortfolioDescription](src/portfolioDescription/aPortfolioDescription.py). 

[Simulator](src/simulator/simulator.py) allows a parallel simulation of several recommending [portfolios](src/portfolio/aPortfolio.py).
It gets on the input a list of child instances of the class [APortfolioDescription](src/portfolioDescription/aPortfolioDescription.py), list of instances
 of portfolio data models and a list of evaluation [operators](src/evaluationTool/aEvalTool.py). During seqential evaluation, evaluation operators iteratively
update portfolio data model representing votes assigned to base [recommenders](src/recommender/aRecommender.py).

![architecture](doc/architecture.png "Visualisation of the architecture")


## Tutorial:

### Installation
- requires Python 3.7

```sh
$ pip install numpy pandas sklearn tensorflow
```

### How to run it:

Before the first run of the system, it si necessary to generate input Batches (task descriptions), which represent simulations and copy them to the input directory.

```sh
$ git clone https://github.com/sbalcar/HeterRecomPortfolio.git
$ cd HeterRecomPortfolio
$ ./generateBatches.sh
$ ./generateBehaviours.sh
$ cp batches/testBatch inputs/
$ ./run.sh
```

The systems in each run takes one file (Batch instance) from the input directory, deletes it and runs it.


#### How to add new Recommender:
- create new child class of [ARecommender](src/recommender/aRecommender.py)
- we recommend adding recommender definition with default parameters to the class [InputRecomDefinition](src/input/inputRecomDefinition.py)

#### How to add new Agregator:
- create new child class of [AAgreggation](src/portfolio/aPortfolio.py)
- we recommend adding aggregator definition with default parameters to the class [InputAggrDefinition](src/input/inputAggrDefinition.py)

#### How to add new Portfolio Architecture:
- create new child class of [APortfolio](src/portfolio/aPortfolio.py)
- after that is necessary to create corresponding descriptive child class of [APortfolioDescription](src/portfolioDescription/aPortfolioDescription.py)
- it's necessary to extends child classes of [ASequentialSimulator](src/simulation/aSequentialSimulation.py) with capability to process this new portfolio architecture

#### How to add new input Batch definition:
- create new child class of [ABatch](src/input/ABatch.py)
- we recommend adding a call to this class' function generateBatches into file [generateBatches](src/portfolio/generateBatches.py)

#### How to add new dataset:
- adding the new dataset requires changing child class of [ARecommender](src/recommender/aRecommender.py), adding new [batches](src/input/ABatch.py) and update of [simulation](src/simulation/aSequentialSimulation.py)


