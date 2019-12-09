import os,logging
from collections import defaultdict

class PPMIModel(object):
    """
    Implementation of a positive pointwise mutual information semantic model.
    """
    def __init__(self,corpus_file,output_path,word_list,L=2,k=1,alpha=0.75,weighting='unweighted'):
        """
        INPUT:
        ------
            corpus_file: string, required
                should be the name of a text file with one document per newline; word-context counting
                is done on a per-document basis

            output_path: string, required
                location to save model output - word/context vectors, raw counts, and computed similarities
                for all pairs of words in the word_list

            word_list: list, required
                any words not in the wordlist will be ignored both for purposes of producing word vectors
                and calculating context; this has no effect on the eventual similarities and drastically
                reduces the amount of space necessary to train the model

            L: integer, optional
                half-window size (full context window will be of size 2*L + 1, symmetric about the center
                word)

            k: integer, optional
                shift parameter; reported ppmi will equal max(ppmi - log(k),0.0); set to 1 for no shift

            alpha: float, optional
                context distribution smoothing parameter (set to 1.0 for no smoothing)

            weighting: string, one of 'unweighted','glove','word2vec'
                'unweighted': words are equally weighted in a fixed length window of 2*L + 1

                'glove' : words are weighted by their distance from the center word divided
                    by L - so for L = 2, weights are [1/2,1,-,1,1/2] (- is the center word)

                'word2vec' : word to vec uses variable length windows (half window size ~ U(1,L))
                    and weights "inverse harmonically"; for L = 2, weights are [1,1/2,-,1/2,1]
        """
        # set the context map
        self.context_map = {'unweighted':self.unweighted_context,'glove':self.glove_context,'word2vec':self.word2vec_context}
        self.corpus_file = corpus_file
        self.output_path = output_path
        # if the output_path does not exist, create it
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)
        self.word_list = word_list
        self.hyperp = {}
        self.hyperp['L'] = L
        self.hyperp['k'] = k
        self.hyperp['alpha'] = alpha
        self.hyperp['weighting'] = weighting
        # custom logging module
        self.logger = logging.getLogger(__name__)
        self.handler = logging.StreamHandler()
        self.formatter = logging.Formatter('%(asctime)s : %(name)s : %(message)s')
        self.handler.setFormatter(self.formatter)
        self.logger.addHandler(self.handler)
        self.logger.setLevel(logging.INFO)            


    def train(self):
        """
        Trains the model on the text in the corpus, using the current values of the
        hyperparameters.
        """
        self.logger.info('Starting model')
        self.logger.info('Writing to %s' % self.output_path)

        # do the pair counting
        wc_counts = self.count_pairs()

        # calculate ppmi
        ppmi = self.calculate_ppmi(wc_counts)

        # compute all pairwise similiarities from the ppmi vectors
        sims = self.calculate_sims(ppmi)

        self.logger.info('Model complete!')

        return sims


    def count_pairs(self):
        """
        Does (w,c) pair counting for the corpus and produces a dictionary of counts, keyed
        on (w,c).
        """
        n_docs = 0
        wc_counts = defaultdict(float)
        with open(self.corpus_file) as f:
            for line in f:
                n_docs += 1
                # list of tokens
                tokens = line.strip().split()
                # skip stub documents (smaller than the window)
                if len(tokens) > 2*self.hyperp['L'] + 1:
                    # iterate over tokens (positions)
                    for i in range(len(tokens)):
                        word = tokens[i]
                        # don't do anything unless it's one of the words in the wordlist
                        if word in self.word_list:
                            # get the context
                            context = self.context_map[self.hyperp['weighting']](tokens,i,self.hyperp['L'])
                            for c,v in context.items():
                                if c in word_list:
                                    wc_counts[(word,c)] += v
                # write a log message if we've made it through 10000 articles
                if (n_docs % 10000 == 0):
                    self.logger.info("Counted (w,c) pairs in  " + str(n_docs) + " articles")
        self.logger.info("(w,c) pair counting complete!")
        self.logger.info("Number of articles: " + str(n_docs))
        # dump the counts
        pickle.dump(wc_counts,open(os.path.join(self.output_path,'wc-counts.pydb'),'wb'))
        return wc_counts


    def calculate_ppmi(self,wc_counts):
        """
        Calculates ppmi from the (w,c) co-occurrence data.
        """
        ppmi = {}
        Pc = defaultdict(float)
        Pw = defaultdict(float)

        # do one loop through the (w,c) pairs to compute P(c) and P(w) and to
        #   initialize the ppmi dictionary
        for k in wc_counts:
            Pw[k[0]] += wc_counts[k]
            Pc[k[1]] += power(wc_counts[k],self.hyperp['alpha'])
            ppmi[k[0]] = {}

        # normalize Pw and Pc
        sum_Pw = sum(Pw.values())
        sum_Pc = sum(Pc.values())
        for k in Pw:
            Pw[k] = Pw[k]/sum_Pw
        for k in Pc:
            Pc[k] = Pc[k]/sum_Pc

        # now compute PPMI
        sum_Pwc = sum(wc_counts.values())
        for k in wc_counts:
            ppmi_value = (wc_counts[k]/sum_Pwc)/(Pw[k[0]]*Pw[k[1]])
            ppmi_value = max([log(ppmi_value)-log(self.hyperp['k']),0.0])
            if ppmi_value > 0.0:
                ppmi[k[0]][k[1]] = ppmi_value

        self.logger.info("PPMI vectors calculated!")
        # dump the ppmi dict
        pickle.dump(ppmi,open(os.path.join(self.output_path,'ppmi-vecs.pydb'),'wb'))
        return ppmi


    def calculate_sims(self,ppmi):
        """
        Calculates all pairwise similarities for the ppmi word vectors.
        """
        sims = {}
        ppmi_words = list(ppmi.keys())

        for i in range(len(ppmi_words)):
            for j in range(i+1,len(ppmi_words)):
                cos_num = 0.0
                cos_den = 0.0
                w_i = ppmi_words[i]
                w_j = ppmi_words[j]
                # find the set of common contexts between the two words
                c_i = set(ppmi[w_i].keys())
                c_j = set(ppmi[w_j].keys())
                overlap = c_i.intersection(c_j)
                # sum up products over common contexts
                for c in overlap:
                    cos_num += ppmi[w_i][c]*ppmi[w_j][c]
                # compute denominator
                cos_den = sqrt(sum([x**2 for x in ppmi[w_i].values()]))*sqrt(sum([x**2 for x in ppmi[w_j].values()]))
                sims[(w_i,w_j)] = cos_num/cos_den

        self.logger.info("Pairwise similarities calculated!")
        # dump the similarity database as a pickle
        pickle.dump(sims,open(os.path.join(self.output_path,'ppmi-sims.pydb'),'wb'))


    def unweighted_context(self,tokens,pos):
        """
        Returns a dictionary of context word: 1 pairs, centered on pos in tokens.  The
        context will consist of 2L tokens and does not include tokens[pos].
        """
        # left-hand window wraps around to the end of the article
        lh_win = list(range(pos-self.hyperp['L'],pos))
        # right-hand window wraps around to the beginning of the article
        rh_win = [mod(x,len(tokens)) for x in list(range(pos+1,pos+self.hyperp['L']+1))]
        # assemble and return context dictionary - getting weird fails for long L, so
        #   use a try/except block
        return {tokens[x]:1 for x in lh_win+rh_win}


    def glove_context(self,tokens,pos):
        """
        GloVe weights context words using their distance from the center word divided
        by the length of the context window.  So a size 2 window has weights (1/2,1,1,1/2).
        """
        # left-hand window wraps around to the end of the article
        lh_win = list(range(pos-self.hyperp['L'],pos))
        # right-hand window wraps around to the beginning of the article
        rh_win = [mod(x,len(tokens)) for x in list(range(pos+1,pos+self.hyperp['L']+1))]
        full_win = lh_win+rh_win
        harm_wts = [(i+1)/self.hyperp['L'] for i in range(self.hyperp['L'])]
        harm_wts = harm_wts + harm_wts[::-1]
        # assemble and context with harmonic weighting (words near pos receive more weight)
        return {tokens[full_win[i]]:harm_wts[i] for i in range(len(full_win))}


    def word2vec_context(self,tokens,pos):
        """
        Word2Vec uses a variable length window (~U(1,L)) and does 'inverse harmonic'
        weighting.  So if the window ends up being size 2, the weights are (1,1/2,1/2,1)
        (see glove_context for why I am calling this 'inverse harmonic').
        """
        # window size is randomly generated
        L = 1 + randint(self.hyperp['L'])
        # left-hand window wraps around to the end of the article
        lh_win = list(range(pos-L,pos))
        # right-hand window wraps around to the beginning of the article
        rh_win = [mod(x,len(tokens)) for x in list(range(pos+1,pos+self.hyperp['L']+1))]
        full_win = lh_win+rh_win
        harm_wts = [(i+1)/self.hyperp['L'] for i in range(self.hyperp['L'])]
        harm_wts = harm_wts[::-1] + harm_wts
        # assemble and context with harmonic weighting (words near pos receive more weight)
        return {tokens[full_win[i]]:harm_wts[i] for i in range(len(full_win))}
