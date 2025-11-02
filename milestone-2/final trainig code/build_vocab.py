    
import nltk
import pickle
import argparse
from collections import Counter
import json
from pycocotools.coco import COCO

nltk.download('punkt')



class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

def build_vocab(json_path, threshold):
    # Load captions from COCO format JSON
    coco = COCO(json_path)
    counter = Counter()
    
    # Count word frequencies
    for ann_id in coco.anns:
        caption = str(coco.anns[ann_id]['caption']).lower()
        tokens = nltk.tokenize.word_tokenize(caption)
        counter.update(tokens)

    # Filter by threshold and build vocab
    words = [word for word, cnt in counter.items() if cnt >= threshold]
    
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')
    
    for word in words:
        vocab.add_word(word)
        
    return vocab

def main(args):
    vocab = build_vocab(json_path=args.caption_path, threshold=args.threshold)
    vocab_path = args.vocab_path
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print("Total vocabulary size: {}".format(len(vocab)))
    print("Saved the vocabulary wrapper to '{}'".format(vocab_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--caption_path', type=str, 
                        default='data/annotations/captions_train2014.json', 
                        help='path for train annotation file')
    parser.add_argument('--vocab_path', type=str, default='./data/vocab.pkl', 
                        help='path for saving vocabulary wrapper')
    parser.add_argument('--threshold', type=int, default=4, 
                        help='minimum word count threshold')
    args = parser.parse_args()
    main(args)