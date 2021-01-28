#!/usr/bin/python3

import io
import csv
from typing import List

from pandas.core.frame import DataFrame  # class

import pandas as pd

import os


class Events:
    COL_VISIT_ID: str = "visitID"
    COL_USER_ID: str = "userID"
    COL_OBJECT_ID: str = "objectID"
    COL_SESSION_ID: str = "sessionID"
    COL_PAGE_ID: str = "pageID"
    COL_PAGE_TYPE: str = "pageType"
    COL_IMAGES_COUNT: str = "imagesCount"
    COL_TEXT_SIZE_COUNT: str = "textSizeCount"
    COL_LINKS_COUNT: str = "linksCount"
    COL_WINDOW_SIZE_X: str = "windowSizeX"
    COL_WINDOW_SIZE_Y: str = "windowSizeY"
    COL_PAGE_SIZE_X: str = "pageSizeX"
    COL_PAGE_SIZE_Y: str = "pageSizeY"
    COL_OBJECTS_LISTED: str = "objectsListed"
    COL_START_DATE_TIME: str = "startDatetime"
    COL_END_DATE_TIME: str = "endDatetime"
    COL_TIME_ON_PAGE: str = "timeOnPage"
    COL_MOUSE_ClicksCount: str = "mouseClicksCount"
    COL_PAGE_VIEWS_COUNT: str = "pageViewsCount"
    COL_MOUSE_MOVING_TIME: str = "mouseMovingTime"
    COL_MOUSE_MOVING_DISTANCE: str = "mouseMovingDistance"
    COL_SCROLLING_COUNT: str = "scrollingCount"
    COL_SCROLLING_TIME: str = "scrollingTime"
    COL_SCROLLING_DISTANCE: str = "scrollingDistance"
    COL_PRINT_PAGE_COUNT: str = "printPageCount"
    COL_SELECT_COUNT: str = "selectCount"
    COL_SELECTED_TEXT: str = "selectedText"
    COL_SEARCHED_TEXT: str = "searchedText"
    COL_COPY_COUNT: str = "copyCount"
    COL_COPY_TEXT: str = "copyText"
    COL_CLICK_ON_PURCHASE_COUNT: str = "clickOnPurchaseCount"
    COL_PURCHASE_COUNT: str = "purchaseCount"
    COL_FORWARDING_TO_LINK_COUNT: str = "forwardingToLinkCount"
    COL_FORWARDED_TO_LINK: str = "forwardedToLink"
    COL_LOG_FILE: str = "logFile"

    @staticmethod
    def getColNameUserID():
        return Events.COL_USER_ID

    @staticmethod
    def getColNameItemID():
        return Events.COL_OBJECT_ID

    @staticmethod
    def readFromFile():
        eventsFile: str = ".." + os.sep + "datasets" + os.sep + "slantour" + os.sep + "new_implicit_events.csv"

#        eventsDF: DataFrame = pd.read_csv(eventsFile, sep=',', usecols=range(35), header=0, encoding="ISO-8859-1",
#                                          engine='python')

        eventsDF: DataFrame = pd.read_csv(eventsFile, sep=',', usecols=[1, 2, 3, 5, 14], header=0,
                                          encoding="ISO-8859-1", engine='python')

        eventsDF[Events.COL_USER_ID] = eventsDF[Events.COL_USER_ID].astype(int)
        eventsDF[Events.COL_OBJECT_ID] = eventsDF[Events.COL_OBJECT_ID].astype(int)
        eventsDF[Events.COL_SESSION_ID] = eventsDF[Events.COL_SESSION_ID].astype(int)

        return eventsDF