# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 13:11:28 2022

@author: "Petalinkar Saša"

Constrstion of training sets for sentimant polarity from Serbain Wordnet

"""
import requests
from srpskiwordnet import SrbSynset
from srpskiwordnet import SrbWordNetReader
import pandas as pd
from nltk.corpus import sentiwordnet as swn
from sklearn.base import BaseEstimator, TransformerMixin

POLARITY = ["POS", "NEG", "OBJ"]


def getObjectiveIDfromWNop(wordnet_corpus_reader:SrbWordNetReader):
    """
    Get sysnst that exist in slected Serbian Wordent and are marked objective
    in WN-op corpus.

    Parameters
    ----------
    wordnet_corpus_reader : SrbWordNetReader
        Reder to Serbain Wornet corpus from which we take sysstes

    Returns
    -------
    Set of Srbsynst
        Set of objective serbain systens

    """
    IDs = list()
    URL = "https://raw.githubusercontent.com/aesuli/SentiWordNet/master/data/Micro-WNop-WN3.txt"
    response = requests.get(URL)
    data = response.text.split('\n')
    common = data[7:115]
    group1= data[118:597]
    group2= data[600:1068]
    for s in common:
        pom = s.split("\t")
        if (pom[0]=="0" and pom[1]=="0"):
            IDs.append(syn2ID(pom[2]))
    for s in group1:
        pom = s.split("\t")
        if (pom[0]=="0" and pom[1]=="0"and pom[2]=="0"and pom[3]=="0"
            and pom[4]=="0"and pom[5]=="0"):
            IDs.append(syn2ID(pom[6]))   
    for s in group2:
        pom = s.split("\t")
        if (pom[0]=="0" and pom[1]=="0"and pom[2]=="0"and pom[3]=="0"):
            IDs.append(syn2ID(pom[4]))        
            
    return set(map(wordnet_corpus_reader.synset_from_ID, IDs))

def getPositiveIDfromWNop(wordnet_corpus_reader:SrbWordNetReader):
    """
    Get sysnst that exist in slected Serbian Wordent and are marked positive
    in WN-op corpus.

    Parameters
    ----------
    wordnet_corpus_reader : SrbWordNetReader
        Reder to Serbain Wornet corpus from which we take sysstes

    Returns
    -------
    Set of Srbsynst
        Set of objective serbain systens

    """
    IDs = list()
    URL = "https://raw.githubusercontent.com/aesuli/SentiWordNet/master/data/Micro-WNop-WN3.txt"
    response = requests.get(URL)
    data = response.text.split('\n')
    common = data[7:115]
    group1= data[118:597]
    group2= data[600:1068]
    for s in common:
        pom = s.split("\t")
        if (pom[0]=="1" and pom[1]=="0"):
            IDs.append(syn2ID(pom[2]))
    for s in group1:
        pom = s.split("\t")
        if (pom[0]=="1" and pom[1]=="0"and pom[2]=="1"and pom[3]=="0"
            and pom[4]=="1"and pom[5]=="0"):
            IDs.append(syn2ID(pom[6]))   
    for s in group2:
        pom = s.split("\t")
        if (pom[0]=="1" and pom[1]=="0"and pom[2]=="1"and pom[3]=="0"):
            IDs.append(syn2ID(pom[4]))        
            
    return set(map(wordnet_corpus_reader.synset_from_ID, IDs))

def getNegativeIDfromWNop(wordnet_corpus_reader:SrbWordNetReader):
    """
    Get sysnst that exist in slected Serbian Wordent and are marked nagative
    in WN-op corpus.

    Parameters
    ----------
    wordnet_corpus_reader : SrbWordNetReader
        Reder to Serbain Wornet corpus from which we take sysstes

    Returns
    -------
    Set of Srbsynst
        Set of objective serbain systens

    """
    IDs = list()
    URL = "https://raw.githubusercontent.com/aesuli/SentiWordNet/master/data/Micro-WNop-WN3.txt"
    response = requests.get(URL)
    data = response.text.split('\n')
    common = data[7:115]
    group1= data[118:597]
    group2= data[600:1068]
    for s in common:
        pom = s.split("\t")
        if (pom[0]=="0" and pom[1]=="1"):
            IDs.append(syn2ID(pom[2]))
    for s in group1:
        pom = s.split("\t")
        if (pom[0]=="0" and pom[1]=="1"and pom[2]=="0"and pom[3]=="1"
            and pom[4]=="0"and pom[5]=="1"):
            IDs.append(syn2ID(pom[6]))   
    for s in group2:
        pom = s.split("\t")
        if (pom[0]=="0" and pom[1]=="1"and pom[2]=="0"and pom[3]=="1"):
            IDs.append(syn2ID(pom[4]))        
            
    return set(map(wordnet_corpus_reader.synset_from_ID, IDs))

def getObjectiveIDfromWSWN(wordnet_corpus_reader:SrbWordNetReader):
    """
    Get sysnst that exist in slected Serbian Wordent and are marked objective
    in SentiWordNet corpus.

    Parameters
    ----------
    wordnet_corpus_reader : SrbWordNetReader
        Reder to Serbain Wornet corpus from which we take sysstes

    Returns
    -------
    Set of Srbsynst
        Set of objective serbain systens

    """
    syns = list()
    URL = "https://raw.githubusercontent.com/aesuli/SentiWordNet/master/data/SentiWordNet_3.0.0.txt"
    response = requests.get(URL)
    data = response.text.split('\n')
    for line in data:
        if line.startswith("#"):
            continue
        pom = line.split("\t")
        if len(pom)<4:
            continue
        pos = pom[0]
        offset = pom[1]
        pos_score = pom[2]
        neg_score = pom[3]
        if pos_score=="0" and neg_score=="0":
            syn_id = syn2ID(pos+ offset)
            syn = wordnet_corpus_reader.synset_from_ID(syn_id)
            if syn is not None:
                if syn.is_definition_in_serbain():
                    syns.append(syn)    
    return syns

def getPositiveIDfromWSWN(wordnet_corpus_reader:SrbWordNetReader):
    """
    Get sysnst that exist in slected Serbian Wordent and are marked objective
    in SentiWordNet corpus.

    Parameters
    ----------
    wordnet_corpus_reader : SrbWordNetReader
        Reder to Serbain Wornet corpus from which we take sysstes

    Returns
    -------
    Set of Srbsynst
        Set of objective serbain systens

    """
    syns = list()
    URL = "https://raw.githubusercontent.com/aesuli/SentiWordNet/master/data/SentiWordNet_3.0.0.txt"
    response = requests.get(URL)
    data = response.text.split('\n')
    for line in data:
        if line.startswith("#"):
            continue
        pom = line.split("\t")
        if len(pom)<4:
            continue
        pos = pom[0]
        offset = pom[1]
        pos_score = pom[2]
        neg_score = pom[3]
        if pos_score=="1" and neg_score=="0":
            syn_id = syn2ID(pos+ offset)
            syn = wordnet_corpus_reader.synset_from_ID(syn_id)
            if syn is not None:
                if syn.is_definition_in_serbain():
                    syns.append(syn)    
    return syns
def getNegativeIDfromWSWN(wordnet_corpus_reader:SrbWordNetReader):
    """
    Get sysnst that exist in slected Serbian Wordent and are marked objective
    in SentiWordNet corpus.

    Parameters
    ----------
    wordnet_corpus_reader : SrbWordNetReader
        Reder to Serbain Wornet corpus from which we take sysstes

    Returns
    -------
    Set of Srbsynst
        Set of objective serbain systens

    """
    syns = list()
    URL = "https://raw.githubusercontent.com/aesuli/SentiWordNet/master/data/SentiWordNet_3.0.0.txt"
    response = requests.get(URL)
    data = response.text.split('\n')
    for line in data:
        if line.startswith("#"):
            continue
        pom = line.split("\t")
        if len(pom)<4:
            continue
        pos = pom[0]
        offset = pom[1]
        pos_score = pom[2]
        neg_score = pom[3]
        if pos_score=="0" and neg_score=="1":
            syn_id = syn2ID(pos+ offset)
            syn = wordnet_corpus_reader.synset_from_ID(syn_id)
            if syn is not None:
                if syn.is_definition_in_serbain():
                    syns.append(syn)    
    return syns


def syn2ID(syn):
    syn = syn.strip()
    ret = "ENG30-" + syn[1:] + "-" + syn[0] 
    return ret


def syn2gloss(syn):
    """
        The syn2gloss function takes a synset as input and returns its definition as a string.
        
        Parameters:
        
        syn: The synset for which the definition is required.
        Returns:
        
        str: The definition of the synset as a string.
    """
    return syn.definition()

def syn2ID2(syn):
    """
    Returns the identifier of a given WordNet Synset.
    
    Parameters:
    syn (WordNet Synset): A WordNet Synset
    
    Returns:
    str: A string representing the identifier of the Synset
    
    """
    return syn.ID()
    
class SrbSynset2GlossTransformer(TransformerMixin, BaseEstimator):
    """
    Class that tranform synst to their gloss.
    """

    def transform(self, X):
        return X.apply(syn2gloss)
  
    def fit(self, X, y):
        return self

# class PolarityDict():
#     def __init__(self, file):
#         head = ["Word", "POS", "NEG"]
#         self.table = pd.read_csv(file, names=head, sep=";")


class PolaritySets():
    """
    A class that stores sets of synsets marked by sentiment from WordNet. 
    Sentiment is divided into objective, positive, and negative. Positive and negative sets 
    are expanded in each iteration by relations between synsets.

    Attributes
    ----------
    _wordnet_corpus_reader : SrbWordNetReader
        A Serbian WordNet corpus reader object.
    _pos : set
        A set of synsets marked as positive.
    _neg : set
        A set of synsets marked as negative.
    _obj : set
        A set of synsets marked as objective.
    _k : int
        The number of iterations performed.

    Methods
    -------
    __init__(self, wordnet_corpus_reader, k=0)
        Initializes a PolaritySets object.
    addPOS(self, word)
        Adds synsets that contain a given lemma to the positive set.
    addPOSIDall(self, IDs)
        Adds all synsets with given IDs to the positive set.
    addPOSID(self, ID)
        Adds a synset with a given ID to the positive set.
    addPOSall(self, words)
        Adds all synsets that contain given lemmas to the positive set.
    addNEG(self, word)
        Adds synsets that contain a given lemma to the negative set.
    addNEGall(self, words)
        Adds all synsets that contain given lemmas to the negative set.
    addNEGID(self, ID)
        Adds a synset with a given ID to the negative set.
    addNEGIDall(self, IDs)
        Adds all synsets with given IDs to the negative set.
    addOBJSYN(self, syns)
        Adds a given set of synsets to the objective set.
    addPOSSYN(self, syns)
        Adds a given set of synsets to the positive set.
    addNEGSYN(self, syns)
        Adds a given set of synsets to the negative set.
    addWNop(self)
        Adds all synsets from WN-op corpus to the objective set.
    addWNopAll(self)
        Adds all synsets from WN-op corpus to the objective set.
    addWSWN(self)
        Adds all synsets from WSWN corpus to the objective set.
    addWSWNALL(self)
        Adds all synsets from WSWN corpus to the objective, positive, and negative sets.
    removeSyn(self, syn, polarity="OBJ")
        Removes a given synset from the stated polarity set (default is objective).
    removeSynID(self, ID, polarity="OBJ")
        Removes a synset with a given ID from the stated polarity set (default is objective).
    removeSynIDs(self, IDs, polarity="OBJ")
        Removes synsets with given IDs from the stated polarity set (default is objective).
    next_itteration(self)
        Returns a new PolaritySets object that has been updated with a new iteration.
    _expandPolarty(self)
        Expands the positive and negative sets based on the relationships between synsets.
    """
    def __init__(self, wordnet_corpus_reader:SrbWordNetReader, k = 0 ):
        """
        Initite sets of synstes marked by sentiment from wordent.

        Setiment is divided in objective, positive and negative.
        Positive and negative set expand in each iterration by relations
        between synstes.

        Parameters
        ----------
        wordnet_corpus_reader : SrbWordNetReader
            DESCRIPTION.
        k : Integer, optional
            NUmber of itteration. The default is 0.

        Returns
        -------
        None.

        """
        self._wordnet_corpus_reader = wordnet_corpus_reader
        self._pos = set()
        self._neg = set()
        self._obj = set()
        self._k = k

    def addPOS(self, word):
        """
        Add sysnsted that contain lemma to posietive set.

        Parameters
        ----------
        word : String
            Lemma

        Returns
        -------
        None.

        """
        syns = self._wordnet_corpus_reader.synsets(word)
        for syn in syns:
            self.addPOSID(syn._ID)            
    def addPOSIDall(self, IDs):
        """
        Add all synsets which ids are in iterate.

        Parameters
        ----------
        IDs : List
            List of IDs

        Returns
        -------
        None.

        """
        for ID in IDs:
            self.addPOSID(ID)

    def addPOSID(self, ID):
        syn = self._wordnet_corpus_reader.synset_from_ID(ID)
        if syn.is_definition_in_serbain():
            self._pos.add(syn)
            self._obj.discard(syn)

    def addPOSall(self, words):
        for word in words:
            self.addPOS(word)

    def addNEG(self, word):
        syns = self._wordnet_corpus_reader.synsets(word)
        for syn in syns:
            self.addNEGID(syn._ID)  

    def addNEGall(self, words):
        for word in words:
            self.addNEG(word)

    def addNEGID(self, ID):
        syn = self._wordnet_corpus_reader.synset_from_ID(ID)
        if syn.is_definition_in_serbain():
            self._neg.add(syn)
            self._obj.discard(syn)
            
            
    def addNEGIDall(self, IDs):
        for ID in IDs:
            self.addNEGID(ID)
    def addOBJSYN(self, syns):
        for syn in syns:
            if syn is not None:
                if syn.is_definition_in_serbain():
                    if  not (syn in self._neg or syn in self._pos):
                        self._obj.add(syn)

    def addPOSSYN(self, syns):
        for syn in syns:
            if syn is not None:
                if syn.is_definition_in_serbain():
                    self._pos.add(syn)
                    self._obj.discard(syn)

    def addNEGSYN(self, syns):
        for syn in syns:
            if syn is not None:
                if syn.is_definition_in_serbain():
                    self._neg.add(syn)
                    self._obj.discard(syn)

    def addWNop(self):
        self.addOBJSYN(getObjectiveIDfromWNop(self._wordnet_corpus_reader))

    def addWNopAll(self):
        """
        Add all synstets from WN-op corpus.

        Returns
        -------
        None.

        """
        self.addOBJSYN(getObjectiveIDfromWNop(self._wordnet_corpus_reader))
        self.addPOSSYN(getPositiveIDfromWNop(self._wordnet_corpus_reader))
        self.addNEGSYN(getNegativeIDfromWNop(self._wordnet_corpus_reader))

    def addWSWN(self):
        self.addOBJSYN(getObjectiveIDfromWSWN(self._wordnet_corpus_reader))
        
    def addWSWNALL(self):
        self.addOBJSYN(getObjectiveIDfromWSWN(self._wordnet_corpus_reader))
        self.addPOSSYN(getPositiveIDfromWSWN(self._wordnet_corpus_reader))
        self.addNEGSYN(getNegativeIDfromWNop(self._wordnet_corpus_reader))
    def removeSyn (self, syn, polarity = "OBJ"):
        """
        Remove synset from stated polarity set. Default objective
        """
        if (polarity == "OBJ"):
           self._obj.discard(syn)
        elif (polarity == "NEG"):
           self._neg.discard(syn)
        elif (polarity == "POS"):
           self._pos.discard(syn)
    def removeSynID (self, ID, polarity = "OBJ"):
        """
        Remove synset by ID from stated polarity set. Default objective
        """
        syn = self._wordnet_corpus_reader.synset_from_ID(ID)
        self.removeSyn (syn, polarity)
    def removeSynIDs (self, IDs, polarity = "OBJ"): 
        """
        Remove synsets by iterative ID from stated polarity set. Default objective
        """ 
        for ID in IDs:
            self.removeSynID(ID, polarity)
        
    def next_itteration (self):
        ret = PolaritySets(self._wordnet_corpus_reader, self._k +1)
        ret._pos = self._pos.copy()
        ret._neg = self._neg.copy()
        ret._obj = self._obj.copy()
        ret._expandPolarty()
        ret.updateDataFrame()
        return ret

    def _expandPolarty(self):
        rel= {"+","=","^"}
        neg = self._neg.copy()
        pos = self._pos.copy()
        #reversed
        for syn in self._pos:
                for s in syn.antonyms():
                    if s is not None:
                        if s.is_definition_in_serbain():
                            neg.add(s)
        for syn in self._neg:
            for s in syn.antonyms():
                if s is not None:
                    if s.is_definition_in_serbain():
                        pos.add(s)
        #preserved
        for r in rel:
            for syn in self._pos:
                for s in syn._related(r):
                    if s is not None:
                        if s.is_definition_in_serbain():
                            pos.add(s)
            for syn in self._neg:
                for s in syn._related(r):
                    if s is not None:
                        if s.is_definition_in_serbain():
                            neg.add(s)
        pos.discard(None)
        neg.discard(None)
        self._neg =neg
        self._pos =pos

    def _getText(self, pol):
        ret = list()
        if (pol == "POS"):
            syns = self._pos
        if (pol == "NEG"):
            syns = self._neg
        if (pol == "OBJ"):
            syns = self._obj
        for syn in syns:
            el = dict()
            el["ID"] = syn.ID()
            el["POS"], el["NEG"] = syn._sentiment
            el["Lemme"] = ",".join(syn._lemma_names)
            el["Definicija"] = syn.definition()
            el["Vrsta"] = syn._POS
            ret.append(el)
        return ret
    def getDef(self):
        ret = dict()
        for t in POLARITY:
            ret[t] = self._getText(t)
        ret["iteration"] = self._k
        return ret

    def updateDataFrame(self):
        """Update internal dataframe.

        Do it after finishing enting systes manuely.
        It automaticly done after expansion by polarity

        Returns
        -------
        None.

        """
        dfpos = pd.DataFrame(self._pos, columns=["Sysnet"])
        dfpos.insert(0, "POS", 1)
        dfpos.insert(0, "NEG", 0)
        dfpos.insert(0, "OBJ", 0)
        dfneg = pd.DataFrame(self._neg, columns=["Sysnet"])
        dfneg.insert(0, "POS", 0)
        dfneg.insert(0, "NEG", 1)
        dfneg.insert(0, "OBJ", 0)
        dfobj = pd.DataFrame(self._obj, columns=["Sysnet"])
        dfobj.insert(0, "POS", 0)
        dfobj.insert(0, "NEG", 0)
        dfobj.insert(0, "OBJ", 1)
        self._df = pd.concat([dfpos, dfneg, dfobj])

    def getXY(self, polatiry="POS", predprocess=None):
        """
        Return X, y for machine learning.

        Parameters
        ----------
        polatiry : [""POS", "NEG], optional
            By which polarity we order y The default is "POS".

        Returns
        -------
        X : Series of SrbSynset
            Data
        y : Series of [0,1]
            Clalsses. 0 is zx not selected poarity, 1 if it is

        """
        X = self._df["Sysnet"]
        y = self._df[polatiry]
        if predprocess is not None:
            for f in predprocess:
                X= X.apply(f)
        
        return X, y

