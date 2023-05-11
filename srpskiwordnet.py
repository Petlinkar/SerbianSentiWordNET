# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 15:06:18 2022

@author: "Petalinkar Saša"

NLTK interface for Serbian Wordnet

 loaded from XML file 


"""
from xml.etree.ElementTree import Element, SubElement
import xml.etree.cElementTree as ET
from nltk.corpus.reader.wordnet import Lemma
from nltk.corpus.reader.wordnet import Synset
from nltk.corpus.reader.xmldocs import XMLCorpusReader
from itertools import chain, islice
import pandas as pd

######################################################################
# Table of Contents
######################################################################
# - Constants
# - Data Classes
#   - Serbian Lemma
#   - Serbian Synset
# - Serbian WordNet Corpus Reader
######################################################################
# Constants
######################################################################

#: Positive infinity (for similarity functions)
_INF = 1e300

# { Part-of-speech constants
ADJ, ADJ_SAT, ADV, NOUN, VERB = "a", "s", "r", "n", "v"
# }

POS_LIST = [NOUN, VERB, ADJ, ADV]

    #This maps symbols used for relations form SW xml to WN format
    # https://wordnet.princeton.edu/documentation/wninput5wn
REL_MAP = {
        "hypernym":"@",
        "hyponym":"~",
        "eng_derivative": "+",  #all are marked as Derivationally related form
        'holo_member': '#m', 
        'derived-vn':'+',       #all are marked as Derivationally related form
        'holo_member': '#m', 
        'particle':'p',         #not in wn5
        'instance_hypernym':'@i',
        'substanceHolonym':'#s',
        'attribute':'=', 
        'SubstanceMeronym':'%s',
        'verb_group':'$',
        'TopicDomain':';c',
        'usage_domain':';u',
        'similar_to':'&', 
        'category_domain':';r', #not in wn5
        'Entailment':'*', 
        'TopicDomainMember':'-c',
        'holo_part':'#p',
        'holo_portion':'#p',
        'mero_portion':'%p', 
        'mero_member':'%m', 
        'entailment':'*', 
        'partMeronym':'%p', 
        'region_domain':';r', 
        'InstanceHyponym':'~i', 
        'causes':'>',
        'be_in_state':"b",      #not in wn5 
        'RegionDomain':';r',
        'subevent':'e',         #not in wn5
        'pertainym':'\\',
        'derived-pos':'+',      #all are marked as Derivationally related form
        'near_antonym':'!',     #marked as antonym
        'DerivedFromAdjective':'\\',
        'specifiedBy':'~',      #this is the defintion of hyponym
        'substanceMeronym':'%s',
        'derived-gender':'+',   #all are marked as Derivationally related form 
        'also_see':'^',     
        'specificOf':'@',       #this is the defintion of hypernym
        'derived':'+',           #all are marked as Derivationally related form
        None:"?"                #None has appered when all possible 
                                #types were listed. This serves for rror check 
        }

######################################################################
# Data Classes
######################################################################

#Serbian Lemma
class SrbLemma(Lemma):
    """The lexical entry for a single morphological form of a
    sense-disambiguated word, for Serbina WordNet
    
    Create a Lemma from a <xml Element> 
    <LITERAL><SENSE></SENSE><LNOTE /></LITERAL>
    Name in PWN is in form
    <word>.<pos>.<number>.<lemma> since we lack that informarion SWNT
    we will just use literal for name of lemma
    """
    def __init__(self, xmlLemma, synset):
        self._lang = "srp"
        self._synset = synset
        self._wordnet_corpus_reader = synset._wordnet_corpus_reader
        if (xmlLemma.tag != "LITERAL"):
            raise Exception ("Not a word")
        self._name = xmlLemma.text
        def get_single(field):
            ret = None
            xml_chunk = xmlLemma.find(field)    
            if xml_chunk is not None:
                ret = xml_chunk.text
            return ret
        self.__annotations__ = get_single("LNOTE")
        self._sense = get_single("SENSE")
        
    def __repr__(self):
        return "<SrbLemma name:%s synset:%s>" % (self._name, self._synset)

    def __str__(self):
        return "SrbLemma: a is %s, b is %s" % (self.a, self.b)


#Serbian Synset
class SrbSynset(Synset):
    """
    A sysnset from Seribian Wordnet.

    Based on Eurowordnet form
    Rewcoreded as XML
    Contain unqiue ID
    """

    def __init__(self, xmlSynset:Element,wordnet_corpus_reader):
        """
        Initilies synstet from XML element 

        Parameters
        ----------
        xmlSynset : Element
            XML element that conatins full desciption of perticualr synset
        wordnet_corpus_reader : SrbWordNetReader
            Corpuse reader linked XML file with Serbian Wornet

        Raises
        ------
        Exception
            "Not a synset" xml elent is not  a synstet
            "Synset lacks id" xml elemnts lack id fileds or its empnty
            
        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        self._wordnet_corpus_reader = wordnet_corpus_reader
        if (xmlSynset.tag != "SYNSET"):
            raise Exception ("Not a synset")
        self._ID = xmlSynset.find("ID").text
        if (self._ID == ""):
            raise Exception ("Synset lacks id")
        
        #here we load literals. If they are missing we reaise na exeption
        xml_pom = xmlSynset.find("SYNONYM")
        self._lemmas = list()
        self._lemma_names =list()
        if (xml_pom is None):
            raise Exception ("Synset "+ self._ID + " lacks literals")
        for lit in xml_pom.findall("LITERAL"):
            pom = SrbLemma(lit, self)
            self._lemmas.append(pom)
            self._lemma_names.append(pom.name())
        #name of the fisrt literal is assigned name of the synset liken PWN
        self._name = self._ID
        
        self._all_hypernyms = None
        self._rel = dict()
        self._relwn5 = dict()
        for rel in xmlSynset.findall("ILR"):
            self._add_rel(rel)
        #fields with sigle text value     
        def get_single(field):
            ret = None
            xml_chunk = xmlSynset.find(field)    
            if xml_chunk is not None:
                ret = xml_chunk.text
            return ret

        self._POS = get_single("POS")
        self._stamp = get_single("STAMP")
        self._definition = get_single("DEF")
        self._definition_lemma = None
        self._domain = get_single("DOMAIN")
        self._NL = get_single("NL")
        self._BCS= get_single("BCS")
        self._SNOTE = get_single("SNOTE")
        #usage/not present in all synsets
        self._examples = list()
        for us in xmlSynset.findall("USAGE"):
            self._examples.append(us.text)
        #sentiment value- originaly in text. conevert to real (replace , with .)
        # we store sentiment as tuple (POS,NEG)
        def txt2real(text):
            return float(text.replace(",","."))
        sent = xmlSynset.find("SENTIMENT")
        self._sentiment = (txt2real(sent.find("POSITIVE").text),txt2real(sent.find("NEGATIVE").text))        

    def definition(self):
        """Return definition in serbian language."""
        return self._definition

    def examples(self):
        """Return examples in serbian language."""
        return self._examples

    def ID(self):
        """Return uniue ID of this synset.

        Returns
        -------
        String
            unquie ID
        """
        return self._ID
    def _add_rel(self, xmlRel:Element):
        """
        Add releshionship to synset, using part of its xml.

        Parameters
        ----------
        xmlRel : Element
            xml representing releshionship under IRL tag
            example:
                <ILR>ENG30-03297735-n<TYPE>hypernym</TYPE></ILR>
        Returns None
        -------
        None.

        """
        pom_id = xmlRel.text
        pom_type = xmlRel.find("TYPE").text
        if (pom_type not in self._rel.keys()):
            self._rel[pom_type] = set()
        self._rel[pom_type].add(pom_id)
        pom_typewn5 = REL_MAP[pom_type]
        if (pom_typewn5 not in self._relwn5.keys()):
            self._relwn5[pom_typewn5] = set()
        self._relwn5[pom_typewn5].add(pom_id)
        
    def _related(self, relation_symbol, sort=False):
        
        get_synset = self._wordnet_corpus_reader.synset_from_ID
        if relation_symbol not in self._relwn5.keys():
            return []
        pointer_ID = self._relwn5[relation_symbol]
        r = [get_synset(ID) for ID in pointer_ID]
        if sort:
            r.sort()
        return r        
    def get_relations_types(self):
        return list(self._rel.keys())
    def __repr__(self):
        return """<SrbSynset ID:%s>
        "Lemma - %s"
        "Definition- %s"
        
        """ % (self._ID, self._lemmas, self.definition() )
    def lemmas(self):
        """
        Returns the set of lemmas that are part of the synset.
        """
        return self._lemmas
    
    def lemma_names(self):
        """
        Returns the list of lemma names that are part of the synset.
        """
        return self._lemma_names
    
    def antonyms(self):
        """
        Returns a set of antonyms (lemmas with opposite meaning) of the synset.
        """
        return self._related("!")
    
    def derivationally_related_forms(self):
        """
        Returns a set of derivationally related forms (related to the synset by morphological derivation).
        """
        return self._related("+")
    
    def POS(self):
        """
        Returns the part-of-speech (POS) of the synset.
        """
        return self._POS


    def parse_definition(self, parser):
        """
        Parses definion of synstet. 

        Parameters
        ----------
        parser : funtion 
            Parser. 

        Returns
        -------
        None.

        """
        self._definition = parser (self._definition)
    def is_definition_in_serbain(self):
        """
        Checks if defintion of synste is in Serbain.
        
        Some sysnets have defintion temorary copied from Prinston WN. 
        That defiention start with "?"

        Returns
        -------
        Boolean
            true Serbian
            false: English or does not exists

        """
        if self.definition() is None:
            return False
        return not self.definition().startswith('?')     
    
    def _estimateSentiment (self, estimators, preprocessor):
        """
        Calculte sentiment using trained ML estimators 

        Parameters
        ----------
        estimators : (sklearn.base.BaseEstimator, sklearn.base.BaseEstimator)
            A tuple of espimators, first for postive, second for negative 
        preprocessor: function
            A pretprocesor for deinition before appling ML 

        Returns
        -------
        (POS, NEG)

        """
        est_POS, est_NEG = estimators
        tekst = preprocessor(self.definition())
        p = est_POS.predict(tekst)
        n = est_NEG.predict(tekst)
        return (p*(1-n), n(1-p))
        
    def __str__(self):
        return "Synset: ID is %s" % (self._ID)
    def __hash__(self):
        return hash(self._ID)
    
# =============================================================================
# Serbian Wornet Reader class
# =============================================================================


class SrbWordNetReader(XMLCorpusReader):
    """Reader for Serbina Wornet based on XML reader."""

    def __init__(self, root, fileids, wrap_etree=False):
        super().__init__(root, fileids, wrap_etree)
        self._path = fileids
#       Here we initilize  dictonary of synsets, since all synsets in
#       Serbain Wordet have unique id, we will be using that as a key
        self._slex =None
        self._synset_dict = dict()
        #this just quick refence table between names and ids. 
        #Reminded name of synset is the firt literal
        self._synset_name = dict()
#       Here we initilize  dictonary of synsets reashionspits, the key will be 
#       tuple (synset id, relashion symbol) while value would resultnty synset
#       key
        self._synset_rel = dict()

        self.load_synsets()
    def synset_from_ID(self, ID):
        """
        Rerurn synster by its unique ID string from wordnet

        Parameters
        ----------
        ID : String
            Uniqe ID assigned to each synset

        Returns
        -------
        SrbSynset
            Synst with required ID.

        """
        if ID in self._synset_dict.keys():
            return self._synset_dict[ID]
        else:
            return None
    def synset_from_name(self, name):
        ID = self._synset_name[name]
        return self._synset_dict[ID]

    def synsets(self, word, POS=None):
        """
        Load all synsets with a given lemma and part of speech tag.

        If no pos is specified, all synsets for all parts of speech
        will be loaded.
        """
        ret = list()
        for id in self._synset_dict:
            syn = self._synset_dict[id]
            if (word in syn.lemma_names()):
                if POS is not None:
                    if POS!= syn.POS():
                        continue
                ret.append(syn)
        return ret
    def load_synsets(self):

        for i, syn in enumerate(self.xml()):
            try:
                pom = SrbSynset(syn, self)
                self._synset_dict[pom.ID()]= pom
                self._synset_name[pom.name()] = pom.ID()
            except Exception as err:
                print(err, i)
        self.rel_types = set()
        for key in self._synset_dict:
            self.rel_types.update(self._synset_dict[key].get_relations_types())
    def synset_from_pos_and_offset(self, pos, offset):
        ID = "ENG30-" + offset +"-" +pos 
        return self.synset_from_ID(ID)

    def parse_all_defintions (self, parser):
        """
        Deprecated function that should not be used.
        """        
        for id in self._synset_dict:
            syn = self._synset_dict[id]
            syn.parse_definition(parser)
            

    def get_relations_types(self):
            return self.rel_types            
    def morphy(self, form, pos=None):
        """
        Return the base form of the given word form and part-of-speech tag using the 
        loaded lexicon.
        
        Parameters
        ----------
        form : str
            The word form to be analyzed.
        pos : str or None, optional
            The part-of-speech tag for the given word form. If None, the function will
            return all possible analyses for the word form across all available
            part-of-speech tags. Default is None.
        
        Returns
        -------
        str or None
            The base form of the given word form based on the morphological analysis,
            or None if no analyses are available.
        """
        if self._slex is None:
            return None
        if pos is None:
             analyses = chain.from_iterable(self._morphy(form, p) 
                                           for p in self._slex["POS"].unique())

        else:
            analyses = self._morphy(form, pos)
        first = list(islice(analyses, 1))
        if len(first) == 1:
            return first[0]
        else:
            return None
    def _morphy(self, form, pos):
        if self._slex is None:
            return []
        else:
            matches = self._slex.loc[(self._slex["POS"] == pos) & (self._slex["Term"] == form)]
            if len(matches) == 0:
                return []
            else:
                return [matches["Lemma"].iloc[0]]
        

    def sentiment_df(self):
        """
        Returns a Pandas DataFrame containing sentiment information for each synset in the WordNet corpus.
        The DataFrame includes the synset ID, positive and negative sentiment scores, lemma names, and definition.
        The sentiment information is extracted from the synset's _sentiment attribute, which contains a tuple of
        two float values representing the synset's positive and negative sentiment scores, respectively. The
        lemma names are obtained from the synset's _lemma_names attribute and are returned as a comma-separated
        string in the DataFrame. The definition of the synset is obtained using the definition() method of the
        synset object. The function returns the DataFrame.
        """        
        syns_list = list()
        for sifra in self._synset_dict:
            syn = self._synset_dict[sifra]
            el = dict()
            el["ID"] = sifra
            el["POS"], el["NEG"] = syn._sentiment
            el["Lemme"] = ",".join(syn._lemma_names)
            el["Definicija"] = syn.definition()
            syns_list.append(el)
        return  pd.DataFrame(syns_list)
        
    def __repr__(self):
        return str(self._synset_dict)
    def load_lexicon (self, path):
        """
        Load a lexicon from a file located at `path`.
        
        Parameters
        ----------
        path : str
            Path to the file containing the lexicon. The file is expected to be
            tab-separated or space-separated, with three columns named "Term",
            "POS", and "Lemma".
        
        Returns
        -------
        None.
        
        """

        self._slex = pd.read_csv(path, sep = "\t| ", 
                   on_bad_lines='skip', names=["Term", "POS", "Lemma"],
                   engine='python')
        
#".\\resources\\lexiconPOSlat"


