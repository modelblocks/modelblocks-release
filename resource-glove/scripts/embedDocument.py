import sys, argparse, math, numpy as np
from nltk.corpus import stopwords
try:
    stopwords.words('english')
except:
    import nltk
    nltk.download('stopwords')
    from nltk.corpus import stopwords

class Embedding(object):
    def __init__(self, word, sentid, embedding):
        self.word = word
        self.sentid = sentid
        self.embedding = embedding

class EmbeddingMap(object):
    def __init__(self, rm_stopwords=False):
        self.map = {}
        self.dim = None
        self.stopwords = set(stopwords.words('english')+["-LRB-", "-RRB-", ",", "-", ";", ":", "\'", "\'\'", '\"', "`", "``", ".", "!", "?", "*FOOT*"])
        self.rm_stopwords = rm_stopwords
        self.measure = self.cosineDistance
    
    def read(self, path):
        with open(path, 'rb') as f:
            line = f.readline()
            while line:
                src = line.strip().split()
                wrd = src[0]
                v = [float(x) for x in src[1:]]
                v = np.array(v)
                if self.dim is None:
                    self.dim = v.shape[0]
                else:
                    assert v.shape[0] == self.dim, 'Malformed input file. Already read in embedding of dim %d, but recieved an embedding of dim %d' %(self.dims, v.shape[0])
                self.map[wrd] = v
                line = f.readline()

    def embedWord(self, word):
        if not self.rm_stopwords or not word in self.stopwords:
            if word in self.map:
                return self.map[word]
        out = np.empty((self.dim))
        out.fill(np.nan)
        return out

    def embedSent(self, sent, sentid=0):
        words = sent.strip().split()
        out = []
        for w in words:
            out.append(Embedding(w, sentid, self.embedWord(w)))
        return out

    def setMeasure(self, s):
        if s == 'cosDist':
            self.measure = self.cosineDistance
        elif s == 'dist':
            self.measure = self.distance
        else:
            raise ValueError('Measure function "%s" is not supported (choose from "cosDist", "dist").' %s)

    def measurable(self, x):
        return (not self.rm_stopwords or x.word not in self.stopwords) and np.isfinite(x.embedding).all()

    def cosineDistance(self, a, b):
        if type(a) == Embedding:
            a = a.embedding
        if type(b) == Embedding:
            b = b.embedding
        num = np.dot(a,b)
        normA = math.sqrt(np.dot(a,a))
        normB = math.sqrt(np.dot(b,b))
        out = 1 - num/(normA*normB)
        return out

    def distance(self, a, b):
        if type(a) == Embedding:
            a = a.embedding
        if type(b) == Embedding:
            b = b.embedding
        diff = a-b
        out = math.sqrt(np.dot(diff, diff))
        return out

    def windowMeasure(self, window, agg='mean'):
        val = 0
        min = np.inf
        max = -np.inf
        count = 0
        for i in range(len(window)):
            w1 = window[i]
            if self.measurable(w1): 
                for j in range(len(window)):
                    w2 = window[j]
                    if i != j and self.measurable(w2):
                        measure = self.measure(w1, w2)
                        val += measure
                        if measure < min:
                            min = measure
                        if measure > max:
                            max = measure
                        count += 1
        if count == 0:
            return np.nan
        elif agg == 'sum':
            return val
        elif agg == 'mean':
            return val / count
        elif agg == 'min':
            return min
        elif agg == 'max':
            return max
        else:
            raise ValueError('Aggregation function "%s" is not supported (choose from "sum", "mean").' %agg)

    def compareWindow(self, w1, window, agg='mean', na_rep=0):
        if not self.measurable(w1):
            return na_rep
        val = 0
        min = np.inf
        max = -np.inf
        count = 0
        for i in range(len(window)):
            w2 = window[i]
            if self.measurable(w2):
                measure = self.measure(w1, w2)
                val += measure
                if measure < min:
                    min = measure
                if measure > max:
                    max = measure
                count += 1
        if count == 0:
            return na_rep
        elif agg == 'sum':
            return val
        elif agg == 'mean':
            return val / count
        elif agg == 'min':
            return min
        elif agg == 'max':
            return max
        else:
            raise ValueError('Aggregation function "%s" is not supported (choose from "sum", "mean").' %agg)

    def windowedSentenceMeasure(self, embeddings, windowLen=np.inf, agg='mean'):
        val = 0
        min = np.inf
        max = -np.inf
        count = 0
        step = windowLen if windowLen < np.inf else len(embeddings)
        for i in range(0, len(embeddings), step):
            measure = self.windowMeasure(embeddings[i:i+step], agg=agg)
            if np.isfinite(measure):
                val += measure
                if measure < min:
                    min = measure
                if measure > max:
                    max = measure
                count += 1
        if count == 0:
            return np.nan
        elif agg == 'sum':
            return val
        elif agg == 'mean':
            return val / count
        elif agg == 'min':
            return min
        elif agg == 'max':
            return max
        else:
            raise ValueError('Aggregation function "%s" is not supported (choose from "sum", "mean").' %agg)

    def printEmbedding(self, t):
        print(' '.join([t.word,str(t.sentid)] + [str(x) for x in t.embedding]))

    def printEmbeddings(self, toklist):
        for t in toklist:
            self.printEmbedding(t)

    def text2Embeddings(self, textfile):
        if type(textfile) == str:
            fromPath = True
            textfile = file(textfile, 'rb')
        else:
            fromPath = False
    
        headers = ['word', 'sentid'] + ['d%d'%x for x in range(1,self.dim+1)]
        print(' '.join(headers))
   
        sentid = 0
        for l in textfile:
            self.printEmbeddings(self.embedSent(l, sentid))
            sentid += 1

        if fromPath:
            textfile.close()

    def printLineMeasure(self, embeddings, windowLen=np.inf):
        sentid = embeddings[0].sentid
        self.setMeasure('cosDist')
        semCosDistMin = self.windowedSentenceMeasure(embeddings, windowLen=windowLen, agg='min')
        semCosDistMax = self.windowedSentenceMeasure(embeddings, windowLen=windowLen, agg='max')
        semCosDistMean = self.windowedSentenceMeasure(embeddings, windowLen=windowLen, agg='mean')
        semCosDistSum = self.windowedSentenceMeasure(embeddings, windowLen=windowLen, agg='sum')

        self.setMeasure('dist')
        semDistMin = self.windowedSentenceMeasure(embeddings, windowLen=windowLen, agg='min')
        semDistMax = self.windowedSentenceMeasure(embeddings, windowLen=windowLen, agg='max')
        semDistMean = self.windowedSentenceMeasure(embeddings, windowLen=windowLen, agg='mean')
        semDistSum = self.windowedSentenceMeasure(embeddings, windowLen=windowLen, agg='sum')

        print(' '.join([str(x) for x in (sentid, semCosDistMin, semCosDistMax, semCosDistMean, semCosDistSum, semDistMin, semDistMax, semDistMean, semDistSum)]))

    def text2LineMeasures(self, textfile, windowLen=np.inf):
        if type(textfile) == str:
            fromPath = True
            textfile = file(textfile, 'rb')
        else:
            fromPath = False

        headers = ['sentid',
                   'semCosDistMin', 'semCosDistMax', 'semCosDistMean', 'semCosDistSum',
                   'semDistMin', 'semDistMax', 'semDistMean', 'semDistSum'
                  ]
        print(' '.join(headers))

        sentid = 0
        for l in textfile:
            embeddings = self.embedSent(l, sentid)
            self.printLineMeasure(embeddings, windowLen)
            sentid += 1

        if fromPath:
            textfile.close()

    def printTokMeasures(self, embeddings, windowLen=np.inf, agg='mean'):
        sentid = embeddings[0].sentid
        for i in range(len(embeddings)):
            self.setMeasure('cosDist')
            incrSemNACosDistMin = self.compareWindow(embeddings[i], embeddings[0:i], agg='min', na_rep=np.nan)
            incrSemNACosDistMax = self.compareWindow(embeddings[i], embeddings[0:i], agg='max', na_rep=np.nan)
            incrSemNACosDistMean = self.compareWindow(embeddings[i], embeddings[0:i], agg='mean', na_rep=np.nan)
            incrSemNACosDistSum = self.compareWindow(embeddings[i], embeddings[0:i], agg='sum', na_rep=np.nan)

            self.setMeasure('dist') 
            incrSemNADistMin = self.compareWindow(embeddings[i], embeddings[0:i], agg='min', na_rep=np.nan)
            incrSemNADistMax = self.compareWindow(embeddings[i], embeddings[0:i], agg='max', na_rep=np.nan)
            incrSemNADistMean = self.compareWindow(embeddings[i], embeddings[0:i], agg='mean', na_rep=np.nan)
            incrSemNADistSum = self.compareWindow(embeddings[i], embeddings[0:i], agg='sum', na_rep=np.nan)

            self.setMeasure('cosDist')
            incrSemCosDistMin = incrSemNACosDistMin if np.isfinite(incrSemNACosDistMin) else 0
            incrSemCosDistMax = incrSemNACosDistMax if np.isfinite(incrSemNACosDistMax) else 0
            incrSemCosDistMean = incrSemNACosDistMean if np.isfinite(incrSemNACosDistMean) else 0
            incrSemCosDistSum = incrSemNACosDistSum if np.isfinite(incrSemNACosDistSum) else 0

            self.setMeasure('dist') 
            incrSemDistMin = incrSemNADistMin if np.isfinite(incrSemNADistMin) else 0
            incrSemDistMax = incrSemNADistMax if np.isfinite(incrSemNADistMax) else 0
            incrSemDistMean = incrSemNADistMean if np.isfinite(incrSemNADistMean) else 0
            incrSemDistSum = incrSemNADistSum if np.isfinite(incrSemNADistSum) else 0

            print(' '.join([str(x) for x in (embeddings[i].word,
                                             embeddings[i].sentid,
                                             incrSemCosDistMin, incrSemCosDistMax, incrSemCosDistMean, incrSemCosDistSum,
                                             incrSemDistMin, incrSemDistMax, incrSemDistMean, incrSemDistSum,
                                             incrSemNACosDistMin, incrSemNACosDistMax, incrSemNACosDistMean, incrSemNACosDistSum,
                                             incrSemNADistMin, incrSemNADistMax, incrSemNADistMean, incrSemNADistSum,
                                             )]))

    def text2TokMeasures(self, textfile, windowLen=np.inf):
        if type(textfile) == str:
            fromPath = True
            textfile = file(textfile, 'rb')
        else:
            fromPath = False

        headers = ['word', 'sentid',
                   'incrSemCosDistMin', 'incrSemCosDistMax', 'incrSemCosDistMean', 'incrSemCosDistSum',
                   'incrSemDistMin', 'incrSemDistMax', 'incrSemDistMean', 'incrSemDistSum',
                   'incrSemNACosDistMin', 'incrSemNACosDistMax', 'incrSemNACosDistMean', 'incrSemNACosDistSum',
                   'incrSemNADistMin', 'incrSemNADistMax', 'incrSemNADistMean', 'incrSemNADistSum',
                  ]
        print(' '.join(headers))
 
        sentid = 0
        for l in textfile:
            embeddings = self.embedSent(l, sentid)
            self.printTokMeasures(embeddings, windowLen)
            sentid += 1
        
        if fromPath:
            textfile.close()


         

def main():
    argparser = argparse.ArgumentParser('''
    Uses a pre-trained embedding map to embed the words in an input document.
    Outputs a data table with one word embedding per line.
    ''')
    argparser.add_argument('embeddings', help='Path to saved word embeddings')
    argparser.add_argument('-t', '--text', default=sys.stdin, help='Path to text file to embed.')
    argparser.add_argument('-s', '--rmStopWrds', action='store_true', help='Don\'t embed stop words (columns will be filled with "nan").')
    argparser.add_argument('-o', '--outputType', default='embeddings', help='Output type (one of: "embeddings", "tokmeasures", "linemeasures").')
    argparser.add_argument('-w', '--windowLen', default='inf', help='Window length in words within which to compute aggregate semantic distance, or "inf" to use a single window per sentence.')
    args, unknown = argparser.parse_known_args()
    if args.windowLen == 'inf':
        args.windowLen = np.inf
    else:
        args.windowLen = int(args.windowLen)

    textfile = args.text
    
    e = EmbeddingMap(args.rmStopWrds)
    e.read(args.embeddings)

    if args.outputType == 'embeddings':
        e.text2Embeddings(textfile)
    elif args.outputType == 'tokmeasures':
        e.text2TokMeasures(textfile, windowLen=args.windowLen)
    elif args.outputType == 'linemeasures':
        e.text2LineMeasures(textfile, windowLen=args.windowLen)

main()
