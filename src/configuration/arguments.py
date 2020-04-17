#!/usr/bin/python3

from typing import List

from configuration.argument import Argument #class


class Arguments:

    def __init__(self, arguments:List[Argument]):
 
        if type(arguments) is not list :
            raise ValueError("Type of argument arguments is not list.")

        #argumentI:Argument
        for argumentI in arguments:
           if type(argumentI) is not Argument:
              raise ValueError("Argument arguments don't contains Argument.")

        self._arguments = arguments;

    def exportArgument(self, name:str):

        if type(name) is not str :
            raise ValueError("Type of argument name is not str.")

        #argumentI:Argument
        for argumentI in self._arguments:
           if argumentI.name == name:
              return argumentI;

        return None;

    def exportArgumentValue(self, name:str):
        # arg:Argument
        arg = self.exportArgument(name)
        
        if arg != None:
          return arg.value

        return None;

    def exportNames(self):
        strings = []
        for argumentI in self._arguments:
           strings.append(argumentI.name);
        #strings:list
        return strings;
          
    def exportValues(self):
        strings = []
        for argumentI in self._arguments:
           strings.append(argumentI.value);
        #strings:list
        return strings;
          
