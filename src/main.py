#!/usr/bin/python3

import time
import sys

from jobs.job01 import job01 #function

def main():

  start = time.time()

  divisionDatasetPercentualSize:int = 50
  job01(divisionDatasetPercentualSize)

  end = time.time()

  print()
  print("Time: " + format(end - start, '.5f') + " s")


if __name__ == "__main__":

  main()

