#!/usr/bin/python3

import pandas as pd


class EvaluationOfRecommenders:

   def __init__(self):
      self._ids:list[str] = []
      self._arguments = []

   def addArg(self, recommenderID:str, argument):
      pass

   def add(self, recommenderID:str, arguments):

      if type(recommenderID) is not str :
         raise ValueError("Argument recommenderID is not type str.")

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
          
       
