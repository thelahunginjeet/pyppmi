import pickle
from numpy import unique

flatten = lambda l: [item for sublist in l for item in sublist]

def read_wordlist_file(file_name):
    """
    Reads a one-word-per-line list of words to accumulate (w,c) counts for.  Duplicates
    are collapsed via a dictionary.
    """
    word_list = {}
    f = open(file_name,'r')
    lines = f.readlines()
    for l in lines:
        word = l.strip()
        if len(word) > 0:
            word_list[word] = None
    return list(word_list.keys())


def read_wordlist_pairs(file_name):
    """
    Constructs a wordlist from a pickled dictionary of pairs of words (i.e.,
    ratings data in which the values are ratings).  The values in the dict are
    ignored; duplicates are handled by initial storage as a dict with None
    values.
    """
    word_list = {}
    r_dict = pickle.load(open(file_name,'rb'))
    for k in r_dict:
        word_list[k[0]] = None
        word_list[k[1]] = None
    return list(word_list.keys())


def aggregate_wordlists(file_names,file_type='pydb'):
    """
    Constructs a master wordlist using multiple files (either text files or pairs
    of ratings).  Each wordlist is constructed, they are joined together, and then
    the unique elements are extracted and the master list is returned.

    file_names should be a list of files to aggregate.
    """
    master_wordlist = []
    if file_type is 'pydb':
        for f in file_names:
            wl = read_wordlist_pairs(f)
            master_wordlist.append(wl)
    # extract and return uniques after flattening the list
    return list(unique(flatten(master_wordlist)))


def lowercase(word_list):
    """
    Simple wrapper to convert all words in the wordlist to lowercase.
    """
    return [x.lower() for x in word_list]
