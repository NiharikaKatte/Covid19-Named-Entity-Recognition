import os
import time
import argparse
import logging
import random
import numpy as np
import pandas as pd
import pickle as pkl
from tqdm import tqdm, trange

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from utils import NerProcessor, get_Dataset, get_glove_word2id_and_word2vec_and_matrix
from models import BiLSTM_CRF
from evaluate import calc_accuracy



logger = logging.getLogger(__name__)

# set the random seed for repeat
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def evaluate(args, data, model, tag_ids):
    model.eval()
    sampler = SequentialSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=args.eval_batch_size)

    logger.info("***** Running eval *****")
    logger.info(f" Num examples = {len(data)}")
    logger.info(f" Batch size = {args.eval_batch_size}")
    
    eval_loss = 0
    nb_eval_steps = 0
    pred_labels = []

    for b_i, (input_ids, input_mask, label_ids) in enumerate(tqdm(dataloader, desc="Evaluating")):
        
        input_ids = input_ids.to(args.device)
        input_mask = input_mask.to(args.device)
        label_ids = label_ids

        with torch.no_grad():
            loss, prediction = model(input_ids, input_mask, label_ids)
        eval_loss += loss.item()
        pred_labels.extend(prediction)
        nb_eval_steps += 1
    
    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = calc_accuracy(pred_labels, tag_ids)
    result = {"eval_loss": eval_loss, "eval_accuracy": eval_accuracy}
    
    output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
    # write eval results to txt file.
    with open(output_eval_file, "a") as writer:
        logger.info("***** Eval results *****")
        logger.info("eval_loss = %.4f", eval_loss)
        logger.info("eval_accuracy = {:.2%}".format(eval_accuracy))
        writer.write("eval_loss = %s\n" % str(round(eval_loss, 4)))
        writer.write("eval_accuracy = %s\n" % (str(round(eval_accuracy*100, 2))+'%'))
        
    return result



def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str)
    parser.add_argument("--glove_txt_file", default=None, type=str)
    parser.add_argument("--output_dir", default=None, type=str)
    parser.add_argument("--checkpoint", default=None, type=str)

    # training parameters
    parser.add_argument("--max_seq_length", default=128, type=int)
    parser.add_argument("--do_train", default=False, type=boolean_string)
    parser.add_argument("--do_eval", default=False, type=boolean_string)
    parser.add_argument("--do_test", default=False, type=boolean_string)
    parser.add_argument("--train_batch_size", default=64, type=int)
    parser.add_argument("--eval_batch_size", default=64, type=int)
    parser.add_argument("--learning_rate", default=1e-3, type=float)
    parser.add_argument("--num_train_epochs", default=100, type=float)
    parser.add_argument("--seed", type=int, default=2019)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument("--max_train_steps", default=-1, type=int)
    parser.add_argument("--evaluate_steps", default=100, type=int)
    parser.add_argument("--patience", default=5, type=int)
    
    parser.add_argument("--rnn_type", type=str, choices=["lstm", "gru"], default="lstm", required=False)
    parser.add_argument("--rnn_layers", default=1, type=int)
    parser.add_argument("--rnn_dim", default=128, type=int)
    parser.add_argument("--rnn_dropout", default=0.0, type=float)

    args = parser.parse_args()
    
    device = torch.device("cuda")
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_
    args.device = device
    
    # set the random seed
    set_seed(args)

    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO)

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    processor = NerProcessor()
    label_map = processor.get_label_map(args)
    num_labels = len(label_map)
    encoded_word2id, _, encoded_matrix = get_glove_word2id_and_word2vec_and_matrix(data_dir=args.data_dir,
                                                                      glove_txt_file=args.glove_txt_file, 
                                                                      output_dir=args.output_dir)
    if args.do_train:

        model = BiLSTM_CRF(weights_matrix=encoded_matrix, 
                           rnn_type=args.rnn_type,
                           num_layers=args.rnn_layers, 
                           rnn_dim=args.rnn_dim, 
                           dropout=args.rnn_dropout, 
                           num_labels=num_labels, 
                           is_trainable=True)
        model.to(device)
        
        train_examples, train_features, train_data = get_Dataset(args, processor, label_map, encoded_word2id, mode="train")
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        if args.do_eval:
            eval_examples, eval_features, eval_data = get_Dataset(args, processor, label_map, encoded_word2id, mode="eval")
            eval_tags = [e.label for e in eval_examples]
            eval_tag_ids = [list(map(lambda x: label_map[x],i)) for i in eval_tags]
      
        if args.max_train_steps > 0:
            t_total = args.max_train_steps
            args.num_train_epochs = args.max_train_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9,0.99))
        
        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_data))
        logger.info("  Num Epochs = %d", args.num_train_epochs)
        logger.info("  Total optimization steps = %d", t_total)

        model.zero_grad()
        global_step = 0
        tr_loss = 0.0
        best_accuracy = 0.0
        patience = 0
        current_eval_loss = 1e8
        
        # added here for reproductibility
        set_seed(args)
        
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            
            epoch_iterator = tqdm(train_dataloader, desc="Training")
            batch_time_avg = 0.0
            
            for step, batch in enumerate(epoch_iterator):
                batch_start = time.time()
                model.train()
                batch = tuple(t.to(args.device) for t in batch)

                loss, _ = model(input_ids=batch[0],
                                input_mask=batch[1],
                                tags=batch[2])
                
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                loss.backward()
                tr_loss += loss.item()
                old_global_step = global_step
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    model.zero_grad()
                    global_step += 1
                    
                    batch_time_avg += time.time() - batch_start
                    description = "Avg. time per gradient updating: {:.4f}s, loss: {:.4f}"\
                                    .format(batch_time_avg/(step+1), tr_loss/global_step)
                    epoch_iterator.set_description(description)
            
                if args.do_eval:
                    if global_step != old_global_step and global_step % args.evaluate_steps == 0:
                        result = evaluate(args, eval_data, model, eval_tag_ids)
                        
                        if result['eval_loss'] < current_eval_loss:
                            current_eval_loss = result['eval_loss']
                            petience = 0
                        else:
                            petience += 1
                        
                        # save the best performs model
                        if result['eval_accuracy'] > best_accuracy and patience <= args.patience:
                            best_accuracy = result['eval_accuracy']
                            now_time = time.strftime('%Y-%m-%d',time.localtime(time.time()))
                            torch.save({"model": model.state_dict(), 
                                        "optimizer": optimizer.state_dict(), 
                                        },
                                       os.path.join(args.output_dir, "model-" + now_time + ".pt"))
                            logger.info("***** Better eval accuracy, save model successfully *****")
                            
                if args.max_train_steps > 0 and global_step > args.max_train_steps:
                    epoch_iterator.close()
                    break
            
            logger.info("Epoch {:} end".format(epoch+1))
            
            if args.max_train_steps > 0 and global_step > args.max_train_steps:
                epoch_iterator.close()
                break
        
        logger.info("End training. After global_step {:}, average_loss = {:.4f}".format(global_step, tr_loss / global_step))
            

    if args.do_test:
        model = BiLSTM_CRF(weights_matrix=encoded_matrix, 
                           rnn_type=args.rnn_type,
                           num_layers=args.rnn_layers, 
                           rnn_dim=args.rnn_dim, 
                           dropout=args.rnn_dropout, 
                           num_labels=num_labels, 
                           is_trainable=True)
        checkpoint = torch.load(args.checkpoint, map_location='cuda:0')
        model.load_state_dict(checkpoint["model"])
        model.to(device)

        test_examples, test_features, test_data = get_Dataset(args, processor, label_map, encoded_word2id, mode="eval")

        logger.info("***** Running test *****")
        logger.info(f" Num examples = {len(test_examples)}")
        logger.info(f" Batch size = {args.eval_batch_size}")

        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.eval_batch_size)
        model.eval()

        pred_labels = []
        
        for b_i, (input_ids, input_mask, _) in enumerate(tqdm(test_dataloader, desc="Predicting")):
            
            input_ids = input_ids.to(args.device)
            input_mask = input_mask.to(args.device)

            with torch.no_grad():
                prediction = model(input_ids, input_mask) # list
                
            pred_labels.extend(prediction)

        id2label = dict(zip(label_map.values(), label_map.keys()))
        pred_labels = [id2label[i] for a in pred_labels for i in a]
        
        test_dict = pkl.load(open(os.path.join(args.data_dir, "val.pkl"), "rb"))
        ids = [str(i)+"_"+str(w_id) for i, _ in enumerate(test_dict["id"]) for w_id, word in enumerate(test_dict["word_seq"][i]) if word !="_w_pad_"]
        
        print(len(ids))
        print(len(pred_labels))
        assert len(ids) == len(pred_labels)
        
        test_preds_dict = {"id":ids, "tag":pred_labels}
        pd.DataFrame(test_preds_dict).to_csv(os.path.join(args.output_dir, "test_pred.csv"), index=False)



if __name__ == "__main__":
    main()
    pass