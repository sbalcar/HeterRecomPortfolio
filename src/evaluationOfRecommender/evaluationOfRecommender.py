#!/usr/bin/python3

from configuration.arguments import Arguments #class
from configuration.argument import Argument #class

from recommender.recommenderDescription import RecommenderDescription #class

from portfolio.portfolioDescription import PortfolioDescription #class


class EvaluationOfRecommenders:

   def __init__(self):
#      if type(portfolioDescr) is not PortfolioDescription :
#         raise ValueError("Argument portfolioDescr is not type PortfolioDescription.")

      self._ids = []
      self._arguments = []

   # recommenderID:str, argument:Argument
   def addArg(self, recommenderID, argument):

      self._ids.append(recommenderID)
      self._arguments.append(Arguments([argument]))

   # recommenderID:str, arguments:Arguments
   def addArgs(self, recommenderID, arguments):

      self._ids.append(recommenderID)
      self._arguments.append(arguments)
