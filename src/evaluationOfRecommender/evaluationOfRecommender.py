#!/usr/bin/python3

from configuration.arguments import Arguments #class
from configuration.argument import Argument #class


class EvaluationOfRecommender:

   def __init__(self):
      self._ids:list[str] = []
      self._arguments:list[Arguments] = []

   def addArg(self, recommenderID:str, argument:Argument):

      self._ids.append(recommenderID)
      self._arguments.append(Arguments([argument]))

   def addArgs(self, recommenderID:str, arguments:Arguments):

      self._ids.append(recommenderID)
      self._arguments.append(arguments)
