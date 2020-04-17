#!/usr/bin/python3

from configuration.arguments import Arguments #class
from configuration.argument import Argument #class


class RecommenderDescription:

    # recommenderClass:Class, arguments:Arguments
    def __init__(self, recommenderClass, arguments:Arguments):

        if type(arguments) is not Arguments :
           raise ValueError("Argument")

        self._recommenderClass = recommenderClass
        self._arguments:Arguments = arguments


    def exportRecommender(self):
        return self._recommenderClass(self._arguments)

