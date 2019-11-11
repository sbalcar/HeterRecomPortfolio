#!/usr/bin/python3

from configuration.argument import Argument #class


class Arguments:

    # arguments:Argument[]
    def __init__(self, arguments):
 
        if type(arguments) is not list :
            raise ValueError("Type of argument arguments is not list.")

        #argumentI:Argument
        for argumentI in arguments:
           if type(argumentI) is not Argument:
              raise ValueError("Argument arguments don't contains Argument.")

        self._arguments = arguments;

    # name:str
    def exportArgument(self, name):

        if type(name) is not str :
            raise ValueError("Type of argument name is not str.")

        #argumentI:Argument
        for argumentI in self._arguments:
           if argumentI.name == name:
              return argumentI;

        return None;

    # name:str
    def exportArgumentValue(self, name):
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
          
