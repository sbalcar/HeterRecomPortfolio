#!/usr/bin/python3

import pandas as pd

from configuration.arguments import Arguments #class
from configuration.argument import Argument #class


class EvaluationOfRecommenders:

   def __init__(self):
      self._ids:list[str] = []
      self._arguments:list[Arguments]= []

   def addArg(self, recommenderID:str, argument:Argument):

      if type(recommenderID) is not str :
         raise ValueError("Argument recommenderID is not type str.")

      if type(argument) is not Argument :
         raise ValueError("Argument argument is not type Argument.")

      self._ids.append(recommenderID)
      self._arguments.append(Arguments([argument]))


   def add(self, recommenderID:str, arguments:Arguments):

      if type(recommenderID) is not str :
         raise ValueError("Argument recommenderID is not type str.")

      if type(arguments) is not Arguments :
         raise ValueError("Argument arguments is not type Arguments.")

      self._ids.append(recommenderID)
      self._arguments.append(arguments)

   def exportAsParamsDF(self):

      if len(self._arguments) == 0:
         methodsParamsDF = pd.DataFrame([], columns=["methodID"])
         methodsParamsDF.set_index("methodID", inplace=True)
         
         return methodsParamsDF;

      methodsParamsData = []
      for indexI in range(len(self._ids)):
         idI = self._ids[indexI];
         argumentsI = self._arguments[indexI];

         listI = [idI] + argumentsI.exportValues()
         methodsParamsData.append(listI)

      columns=["methodID"] + self._arguments[0].exportNames()

      methodsParamsDF = pd.DataFrame(methodsParamsData, columns=columns)
      methodsParamsDF.set_index("methodID", inplace=True)

      return methodsParamsDF;
          
       
