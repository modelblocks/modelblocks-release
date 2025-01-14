"""
@author: Hongao ZHU

@description:
    deduplication based on 'naturalstories.sentitems' and 'naturalstories.unigram.itemmeasures' files

@example usage:
    cd ~/zhu.3986
    conda activate deduplication
    python3 scripts/deduplicate_new.py -B 0_142998 -N 10 -T modelblocks-release/workspace/genmodel/naturalstories.sentitems -D modelblocks-release/workspace/genmodel/naturalstories.unigram.itemmeasures -C /fs/project/schuler.77/corpora/pythia_pile
    python3 scripts/deduplicate_new.py -B 0_142998 -N 10 -T modelblocks-release/workspace/genmodel/geco.sentitems -D modelblocks-release/workspace/genmodel/geco.unigram.itemmeasures -C /fs/project/schuler.77/corpora/pythia_pile

"""

import numpy as np, pandas as pd
import sys, re, heapq, argparse
from transformers import AutoTokenizer

# redirect output into logs
def setup_logging(output_file):
    sys.stdout = open(f'{output_file}.log', 'a')
    sys.stderr = open(f'{output_file}.log', 'a')

tokenizer = AutoTokenizer.from_pretrained(
    "EleutherAI/pythia-70m-deduped",
    revision="step3000",
    cache_dir="./pythia-70m-deduped/step3000",
)

class BatchIterator:
    def __init__(self, start_batch=0, end_batch=142998, data_dir='/fs/project/schuler.77/corpora/pythia_pile'):
        self.data_dir = data_dir
        self.batch_size = 1024
        self.current_index = start_batch
        self.end_index = end_batch
        self.current_file = None
        self.current_data = None
        self.file_index = 0

    def get_batch(self, index):
        file_name = f"{self.data_dir}/batch_{index//1000*1000}_to_{index//1000*1000+1000}.npy"
        batch_start = (index % 1000) * self.batch_size
        batch_end = batch_start + self.batch_size
        # load data using np.memmap
        # mmapped_data = np.memmap(file_name, dtype='uint16', mode='r', shape=(2098176000,))
        mmapped_data = np.load(file_name)
        batch_data = mmapped_data[batch_start:batch_end]
        return batch_data

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_index >= self.end_index:
            raise StopIteration
        self.current_data = self.get_batch(self.current_index)
        self.file_index += 1
        self.current_index += 1
        return self.current_data

def calculate_average_surprisal(file_path):
    word_surprisal_map = {}
    with open(file_path, 'r') as file:
        next(file)  # skip headers
        for line in file:
            word, surprisal = line.strip().split()
            word = re.sub(r'^[^a-zA-Z]+|[^a-zA-Z]+$', '', word)
            surprisal = float(surprisal)
            if word in word_surprisal_map:
                word_surprisal_map[word]['count'] += 1
                word_surprisal_map[word]['total_surprisal'] += surprisal
            else:
                word_surprisal_map[word] = {'count': 1, 'total_surprisal': surprisal}

    return dict(sorted({word: data['total_surprisal'] / data['count'] for word, data in word_surprisal_map.items()}.items(), key=lambda item: item[1]))

def get_highest_surprisal_triples(sentitems_path, unigram_itemmeasures_path, n=10):
    # get dictionary
    surprisal_dict = calculate_average_surprisal(unigram_itemmeasures_path)
    
    # get articles
    with open(sentitems_path, 'r') as f:
        lines = f.readlines()
    articles = []
    current_article = ''

    if "!ARTICLE" in lines: # if the text has seperate marks for articles
        for line in lines:
            if line.strip() == "!ARTICLE":
                if current_article: 
                    articles.append(current_article)
                    current_article = ''
            else:
                current_article+=line.strip()
    else: # if the artile has not seperate marks, separate by 100 lines
        line_count = 0
        for line in lines:
            if line_count < 100:
                current_article += line.strip()
                line_count += 1
            else:
                articles.append(current_article)
                current_article = line.strip()
                line_count = 1  # Reset line count and increment for the current line
        
    if current_article: # for the last article
        articles.append(current_article)
    
    # get the highest surprisal values for each article
    highest_surprisal_triples = []
    for article in articles:
        # get all 3-grams
        words = re.findall(r'\b\w+\b', article)
        triples = [tuple(words[i:i+3]) for i in range(len(words) - 2)]
        surprisals = [sum([surprisal_dict.get(word, 0) for word in triple]) for triple in triples]

        # Use heapq.nlargest to get the n highest surprisal triples
        highest_n_triples = heapq.nlargest(n, zip(triples, surprisals), key=lambda x: x[1])
        highest_surprisal_triples.extend(highest_n_triples)
    
    return [triple[0] for triple in highest_surprisal_triples]

def deduplicate(batch_range, sentitems_path, unigram_itemmeasures_path, n=10, corpora_dir='/fs/project/schuler.77/corpora/pythia_pile'):
    # Get batches and 3-grams
    start_batch, end_batch = map(int, batch_range.split('_'))
    batches = BatchIterator(start_batch, end_batch, corpora_dir)
    ref_triples = get_highest_surprisal_triples(sentitems_path, unigram_itemmeasures_path, n)
    print(f"Searching for triples:{ref_triples}:")

    # Process batches
    records = []
    for i, batch in enumerate(batches):
        print(f"processing batch {i+start_batch}...")
        for j, text in enumerate(batch):  # 1024 texts in one batch
            decoded_text = tokenizer.decode(text)
            words = re.findall(r'\b\w+\b', decoded_text)
            text_triples = [tuple(words[i:i+3]) for i in range(len(words)-2)]
            duplicates = set(text_triples).intersection(ref_triples)

            if duplicates:
                print(f"Found duplicates in Batch: {i+start_batch} Text: {j}.")
                records.append((i+start_batch,j,list(duplicates),decoded_text))

        # Save the surprisal values when one batch is completed
        df = pd.DataFrame(records)
        df.to_csv(f"{batch_range}.csv",index=False)
        del df
    print("Deduplication Complete!!!")

def main():
    # Set up the argument parser
    parser = argparse.ArgumentParser(description='Get surprisal estimation for the Pythia training batches.')
    parser.add_argument('--batch', '-B', type=str, default='0_1', help='Batch range. Default:0_1.')
    parser.add_argument('--num', '-N', type=int, default=10, help='Number of most surprised 3-grams per article.')
    parser.add_argument('--text', '-T', type=str, default='naturalstories.sentitems', help='naturalstories.sentitems')
    parser.add_argument('--dict', '-D', type=str, default='naturalstories.unigram.itemmeasures', help='naturalstories.unigram.itemmeasures')
    parser.add_argument('--corpora', '-C', type=str, default='/fs/project/schuler.77/corpora/pythia_pile', help='/fs/project/schuler.77/corpora/pythia_pile')
    args = parser.parse_args()

    setup_logging(args.batch)
    deduplicate(args.batch,args.text,args.dict,args.num, args.corpora)


if __name__ == '__main__':
    main()
