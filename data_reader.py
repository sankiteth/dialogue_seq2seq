# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


import gzip
import os
import re
import tarfile
import operator

from six.moves import urllib

from tensorflow.python.platform import gfile

# Special vocabulary symbols - we always put them at the start.
_PAD = "_PAD"
_GO = "_GO"
_EOS = "_EOS"
_UNK = "_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

#modified
def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size):
  """Create vocabulary file (if it does not exist yet) from data file.

  Data file has triplets. The 3 conversation within a triplet are separated by '\t', and the triplets are
  separated by '\n'

  Args:
    vocabulary_path: path where the vocabulary will be created.
    data_path: data file that will be used to create vocabulary.
    max_vocabulary_size: limit on the size of the created vocabulary.
  """

  print("In create_vocabulary")
  if not gfile.Exists(vocabulary_path):
    print("Creating vocabulary %s from data %s" % (vocabulary_path, data_path))
    vocab = {}
    with gfile.GFile(data_path, mode="r") as f:
      counter = 0
      for line in f:
        counter += 1
        if counter % 100000 == 0:
          print("processing line %d" % counter)
        text_conversation =line.strip().split("\t")
    
        txt  = text_conversation[0].strip() + " " + text_conversation[1].strip() + " " + text_conversation[2].strip()

        tokens = txt.split()
        for w in tokens:
          word = w
          if word in vocab:
            vocab[word] += 1
          else:
            vocab[word] = 1


      vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
      print("vocab_length={0}".format(len(vocab_list)))

      if len(vocab_list) > max_vocabulary_size:
        vocab_list = vocab_list[:max_vocabulary_size]

      with gfile.GFile(vocabulary_path, mode="w") as vocab_file:
        for w in vocab_list:
          vocab_file.write(w + "\n")

#modified
def initialize_vocabulary(vocabulary_path):
  """Initialize vocabulary from file.

  We assume the vocabulary is stored one-item-per-line, so a file:
    dog
    cat
  will result in a vocabulary {"dog": 0, "cat": 1}, and this function will
  also return the reversed-vocabulary ["dog", "cat"].

  Args:
    vocabulary_path: path to the file containing the vocabulary.

  Returns:
    a pair: the vocabulary (a dictionary mapping string to integers), and
    the reversed vocabulary (a list, which reverses the vocabulary mapping).

  Raises:
    ValueError: if the provided vocabulary_path does not exist.
  """

  print("In initialize_vocabulary")
  if gfile.Exists(vocabulary_path):
    rev_vocab = []
    with gfile.GFile(vocabulary_path, mode="r") as f:
      rev_vocab.extend(f.readlines())
    rev_vocab = [line.strip() for line in rev_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    return vocab, rev_vocab
  else:
    raise ValueError("Vocabulary file %s not found.", vocabulary_path)

#modified
def sentence_to_token_ids(sentence, vocabulary):
  """Convert a string to list of integers representing token-ids.

  For example, a sentence "I have a dog" may become tokenized into
  ["I", "have", "a", "dog"] and with vocabulary {"I": 1, "have": 2,
  "a": 4, "dog": 7"} this function will return [1, 2, 4, 7].

  Args:
    sentence: a string, the sentence to convert to token-ids.
    vocabulary: a dictionary mapping tokens to integers.
    tokenizer: a function to use to tokenize each sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.

  Returns:
    a list of integers, the token-ids for the sentence.
  """
  words = sentence.strip().split()
  return [vocabulary.get(w, UNK_ID) for w in words]

def vocab_experiment():
  vocab = {}
  total = 0
  with gfile.GFile("data/Training_Shuffled_Dataset.txt", mode="r") as f:
    counter = 0
    for line in f:
      counter += 1
      if counter % 100000 == 0:
        print("processing line %d" % counter)
      text_conversation =line.strip().split("\t")
  
      txt  = text_conversation[0].strip() + " " + text_conversation[1].strip() + " " + text_conversation[2].strip()
      tokens = txt.split()
      for w in tokens:
        word = w
        total += 1
        if word in vocab:
          vocab[word] += 1
        else:
          vocab[word] = 1


  print(len(vocab))
  sorted_vocab = sorted(vocab.items(), key=operator.itemgetter(1), reverse=True)
  print(sorted_vocab[0])

  cum = 0
  percentage = 0
  counter = 0
  for item in sorted_vocab:
    counter += 1
    cum += item[1]

    percentage = cum*100.0/total

    if counter == 3000:
      print("counter={0}, percentage={1}%".format(counter, percentage))

def bucket_experiment():
  buckets = [0,0,0,0,0,0]
  total_examples = 0
  with gfile.GFile("data/Training_Shuffled_Dataset.txt", mode="r") as f:
    for line in f:
      text_conversation =line.strip().split("\t")
  
      txt1  = text_conversation[0].strip()
      txt2  = text_conversation[1].strip()
      txt3  = text_conversation[2].strip()

      for item in [(txt1, txt2), (txt2, txt3)]:
        total_examples += 1
        enc = item[0]
        dec = item[1]
        num_enc = len(enc.split())
        num_dec = len(dec.split())

        if num_enc <= 5 and num_dec <= 10:
          buckets[0] += 1
        elif num_enc <= 10 and num_dec <= 15:
          buckets[1] += 1
        elif num_enc <= 20 and num_dec <= 25:
          buckets[2] += 1
        elif num_enc <= 40 and num_dec <= 50:
          buckets[3] += 1
        elif num_enc <= 100 and num_dec <= 100:
          buckets[4] += 1
        else:
          buckets[5] += 1



  print(buckets)
  print("total examples  ={0}".format(total_examples))
  print("examples covered={0}".format(sum(buckets[:5])))

# if __name__ == '__main__':
#   #create_vocabulary("data/Vocab_file.txt", "data/Training_Shuffled_Dataset.txt", 0)
#   #buckets = [[0,0], [0,0], [0,0], [0,0], [0,0], [0,0]]
  

  

