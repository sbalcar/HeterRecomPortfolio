#!/usr/bin/python3

import numpy as np

from recommendation.resultOfRecommendation import ResultOfRecommendation #class


class Recommendation:

   # itemIDs:Int[]
   def __init__(self, itemIDs):
        
       if itemIDs == None :
           raise ValueError("Argument itemIDs can't be None.")
       if type(itemIDs) is not list:
           raise ValueError("Argument itemIDs isn't list.")
       if len(itemIDs) == 0:
           raise ValueError("Argument itemIDs is empty.")

       self.itemIDs = itemIDs;


   def exportAsResultOfRecommendation(self):

      rating = [0.1] * len(self.itemIDs)
      return ResultOfRecommendation(self.itemIDs, rating);

