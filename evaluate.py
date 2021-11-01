import json
import pandas as pd
import pickle as pkl
import numpy as np


# Provided function to test accuracy
# You could check the validation accuracy to select the best of your models
def calc_accuracy(preds, tag_ids):
    """
        Input:
            preds (list): without pad
            tags  (list): without pad
        Output:
            Proportion of correct prediction.
    """
    print(len(preds))
    print(len(tag_ids))
    
    # predicts = []
    # for i in range(len(tag_ids)):
    #     predicts.extend(preds[i][: len(tag_ids[i])])
    predicts = [i for a in preds for i in a]
    tag_ids = [i for a in tag_ids for i in a]
    
    print(len(predicts))
    print(len(tag_ids))
    assert len(predicts) == len(tag_ids)
    
    return sum(np.array(predicts) == np.array(tag_ids)) / len(tag_ids)

def evaluate(pred_file, ground_file):
    file_dict = pkl.load(open(ground_file, "rb"))
    file_preds = pd.read_csv(pred_file)
    return calc_accuracy([json.loads(line) for line in file_preds["labels"]], 
              [i for l in file_dict["tag_seq"] for i in l if i != "_t_pad_"]
              )