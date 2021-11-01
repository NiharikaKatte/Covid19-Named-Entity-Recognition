import argparse
import logging
import os
import numpy as np
import torch
import pickle as pkl

from torch.utils.data import TensorDataset
from tqdm import tqdm, trange

logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, label):
        self.guid = guid
        self.text = text 
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, label_id):

        self.input_ids = input_ids
        self.input_mask = input_mask
        self.label_id = label_id


class NerProcessor(object):
    
    def get_label_map(self, args):
        # get labels from train data
        if os.path.exists(os.path.join(args.output_dir, "label_map.pkl")):
            print("label_map file already existed")
            with open(os.path.join(args.output_dir, "label_map.pkl"), "rb") as f:
                tag_dict = pkl.load(f)
        else:
            logger.info(f"loading labels info from train file and dump in {args.output_dir}")
            train_dict = pkl.load(open(os.path.join(args.data_dir, "train.pkl"), "rb"))
            tag_dict = {}
            for tag_seq in train_dict['tag_seq']:
                for tag in tag_seq:
                    if(tag != '_t_pad_' and tag not in tag_dict):
                        tag_dict[tag] = len(tag_dict)
        
            with open(os.path.join(args.output_dir, "label_map.pkl"), "wb") as f:
                pkl.dump(tag_dict, f)
        
        return tag_dict
    

    def get_examples(self, input_file, is_labeling=True):
        examples = []
        
        data_dict = pkl.load(open(input_file, "rb"))
        
        for i in range(len(data_dict["id"])):
        # for i in range(100):
            guid = data_dict["id"][i]
            text = [t for t in data_dict["word_seq"][i] if t != "_w_pad_"]
            label = [l for l in data_dict["tag_seq"][i] if l != "_t_pad_"] if is_labeling else ['O'] * len(text)
            
            if len(text) > 0:
                examples.append(InputExample(guid=guid, text=text, label=label))
        
        return examples


def convert_examples_to_features(args, examples, label_map, max_seq_length, encoded_word2id):

    features = []

    for (ex_index, example) in tqdm(enumerate(examples), desc="convert examples"):
        input_ids = [encoded_word2id[i] for i in example.text]
        label_ids = [label_map[i] for i in example.label]
        input_mask = [1] * len(input_ids)
        
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            label_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(label_ids) == max_seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join([str(x) for x in [i for i in example.text]]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              label_id=label_ids))
    
    return features


def get_Dataset(args, processor, label_map, encoded_word2id, mode="train"):
    if mode == "train":
        filepath = os.path.join(args.data_dir, "train.pkl")
        is_labeling = True
    elif mode == "eval":
        filepath = os.path.join(args.data_dir, "val.pkl")
        is_labeling = True
    elif mode == "test":
        filepath = os.path.join(args.data_dir, "test.pkl")
        is_labeling = False
    else:
        raise ValueError("mode must be one of train, eval, or test")

    examples = processor.get_examples(filepath, is_labeling=is_labeling)

    features = convert_examples_to_features(
        args, examples, label_map, args.max_seq_length, encoded_word2id
    )
    
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    
    data = TensorDataset(all_input_ids, all_input_mask, all_label_ids)

    return examples, features, data


def get_glove_word2id_and_word2vec_and_matrix(data_dir, glove_txt_file, output_dir):
    
    glove_dimension = int(glove_txt_file.split('.')[-2][:-1]) # './glove/glove.6B.100d.txt'
    glove_size = int(glove_txt_file.split('.')[-3][:-1])
    
    if not os.path.exists(os.path.join(output_dir, "glove/")):
        os.makedirs(os.path.join(output_dir, "glove/"))
    encoded_word2id_path = os.path.join(output_dir, "glove/word2id.pkl")
    encoded_word2vec_path = os.path.join(output_dir, "glove/word2vec_{}B_{}d.npy".format(glove_size, glove_dimension))
    encoded_matrix_path = os.path.join(output_dir, "glove/id2vec_{}B_{}d.npy".format(glove_size, glove_dimension))
    
    if os.path.exists(encoded_word2id_path) \
        and os.path.exists(encoded_word2vec_path) \
        and os.path.exists(encoded_matrix_path): 
            print("These files already existed")
            
            with open(encoded_word2id_path, "rb") as f:
                encoded_word2id = pkl.load(f)
            encoded_word2vec = np.load(encoded_word2vec_path, allow_pickle=True)
            encoded_matrix = np.load(encoded_matrix_path, allow_pickle=True)
    
    else:
        # Load the data for three splits
        train_dict = pkl.load(open(os.path.join(data_dir, "train.pkl"), 'rb'))
        val_dict = pkl.load(open(os.path.join(data_dir, "val.pkl"), 'rb'))
        test_dict = pkl.load(open(os.path.join(data_dir, "test.pkl"), 'rb'))

        glove_dict = {}
        with open(glove_txt_file, 'r', encoding="utf-8") as f:
            for line in tqdm(f):
                values = line.split(' ')
                word = values[0]
                vector = np.asarray(values[1:], "float32")
                glove_dict[word] = vector
            
        # Give all the words appeared in our corpus their glove embedding, for those who are not exist, random initialize them
        encoded_word2vec = {}
        count = 0
        total = 0
        glove_keys = glove_dict.keys()
        for i in [train_dict, val_dict, test_dict]:
            for j in trange(len(i['word_seq'])):
                for word in i['word_seq'][j]:
                    if word not in glove_keys:
                        encoded_word2vec[word] = np.random.rand(1, glove_dimension)[0]
                        count += 1
                        total += 1
                    else:
                        encoded_word2vec[word] = glove_dict[word]
                        total += 1
        # Test how many words are found in glove and how many are randomly initialized
        print("words not found {}".format(count))
        print("words total {}".format(total))
        print(len(encoded_word2vec))
        np.save(encoded_word2vec_path, encoded_word2vec)

        # Build a dict that records the word to a single unique integer, and our encoded matrix for word embedding
        encoded_word2id = {}
        encoded_matrix = np.zeros((len(encoded_word2vec.keys()), glove_dimension), dtype=float)
        for i, word in enumerate(encoded_word2vec.keys()):
            encoded_word2id[word] = i
            encoded_matrix[i] = encoded_word2vec[word]
        print(encoded_matrix.shape)
        np.save(encoded_matrix_path, encoded_matrix)
    
        with open(encoded_word2id_path, "wb") as f:
            pkl.dump(encoded_word2id, f)
        
    return encoded_word2id, encoded_word2vec, encoded_matrix





if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default='C:/Users/31906/Desktop/HKUST/Curriculum/MSBD6000H Natural Language Processing/project2/data/', type=str)
    parser.add_argument("--glove_txt_file", default='C:/Users/31906/Desktop/HKUST/Curriculum/MSBD6000H Natural Language Processing/project2/glove.6B.300d.txt', type=str)
    parser.add_argument("--output_dir", default='C:/Users/31906/Desktop/HKUST/Curriculum/MSBD6000H Natural Language Processing/project2/output/', type=str)
    parser.add_argument("--max_seq_length", default=128, type=int)

    args = parser.parse_args()
    
    processor = NerProcessor()
    label_map = processor.get_label_map(args)
    encoded_word2id, _, _ = get_glove_word2id_and_word2vec_and_matrix(data_dir=args.data_dir,
                                                                      glove_txt_file=args.glove_txt_file, 
                                                                      output_dir=args.output_dir)
    
    examples, features, data = get_Dataset(args, processor, label_map, encoded_word2id, mode="test")
    
###### max_train_length: 598
###### max_val_length: 323
###### max_test_length: 721











