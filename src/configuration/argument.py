#!/usr/bin/python3

class Argument:

    # name:String, value:?
    def __init__(self, name:str, value):

        if name == None :
           raise ValueError("Argument name can't be None.")

        if value == None :
           raise ValueError("Argument value can't be None.")

        self.name = name;
        self.value = value;

