# -*- coding: utf-8 -*-
"""
Created on Mon Oct 03 21:23:51 2016

@author: CHOU_H
"""

import nltk

from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer("english")

print stemmer.stem('response')
print stemmer.stem('resposivity')
print stemmer.stem('unresponsive')