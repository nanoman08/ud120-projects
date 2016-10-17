# -*- coding: utf-8 -*-
"""
Created on Fri Jul 01 11:52:39 2016

@author: CHOU_H
"""

"""Count words."""
from collections import Counter
def count_words(s, n):
    """Return the n most frequently occuring words in s."""
    a = s.split(' ')
    # TODO: Count the number of occurences of each word in s
    b = Counter(a)
    # TODO: Sort the occurences in descending order (alphabetically in case of ties)
    c = sorted(b.iteritems(), key=lambda tup:(-tup[1], tup[0]))
    # TODO: Return the top n words as a list of tuples (<word>, <count>)
    top_n = c[:n]
    return top_n


def test_run():
    """Test count_words() with some inputs."""
    print count_words("cat bat mat cat bat cat", 3)
    print count_words("betty bought a bit of butter but the butter was bitter", 3)

if __name__ == '__main__':
    test_run()
    
sample_memo = '''
Milt, we're gonna need to go ahead and move you downstairs into storage B. We have some new people coming in, and we need all the space we can get. So if you could just go ahead and pack up your stuff and move it down there, that would be terrific, OK?
Oh, and remember: next Friday... is Hawaiian shirt day. So, you know, if you want to, go ahead and wear a Hawaiian shirt and jeans.
Oh, oh, and I almost forgot. Ahh, I'm also gonna need you to go ahead and come in on Sunday, too...
Hello Peter, whats happening? Ummm, I'm gonna need you to go ahead and come in tomorrow. So if you could be here around 9 that would be great, mmmk... oh oh! and I almost forgot ahh, I'm also gonna need you to go ahead and come in on Sunday too, kay. We ahh lost some people this week and ah, we sorta need to play catch up.
'''

#
#   Maximum Likelihood Hypothesis
#
#
#   In this quiz we will find the maximum likelihood word based on the preceding word
#
#   Fill in the NextWordProbability procedure so that it takes in sample text and a word,
#   and returns a dictionary with keys the set of words that come after, whose values are
#   the number of times the key comes after that word.
#   
#   Just use .split() to split the sample_memo text into words separated by spaces.

from collections import defaultdict

def NextWordProbability(sampletext,word):
    
    
    words_after=defaultdict(int)
    test_split = sampletext.split()
    for i in range(len(test_split)-1):
        if test_split[i] == word:
            words_after[test_split[i+1]]+=1
        
    
        
        
    
    return words_after
