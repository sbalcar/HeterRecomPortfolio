#!/usr/bin/python3

import time
import numpy as np
import pandas as pd

#from jobs.recommendation import recommendation #function
from jobs.simulationOfPortfolio import simulationOfPortfolio #function

def main():

  start = time.time()

  ##recommendation()
  simulationOfPortfolio()

  end = time.time()

  print()
  print("Time: " + format(end - start, '.5f') + " s")


if __name__ == "__main__":
    main()

