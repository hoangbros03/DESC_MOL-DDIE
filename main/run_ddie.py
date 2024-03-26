# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa)."""

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import random
import json
import copy

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from tqdm import tqdm, trange

from transformers import (WEIGHTS_NAME, BertConfig,
                                  #BertForSequenceClassification,
                                  BertTokenizer,
                                  RobertaConfig,
                                  RobertaForSequenceClassification,
                                  RobertaTokenizer,
                                  XLMConfig, XLMForSequenceClassification,
                                  XLMTokenizer, XLNetConfig,
                                  XLNetForSequenceClassification,
                                  XLNetTokenizer,
                                  DistilBertConfig,
                                  DistilBertForSequenceClassification,
                                  DistilBertTokenizer)

from transformers import AdamW, WarmupLinearSchedule
from radam import RAdam, PlainRAdam, AdamW

from modeling_ddie import BertForSequenceClassification

#from transformers import glue_compute_metrics as compute_metrics
from metrics_ddie import ddie_compute_metrics as compute_metrics
from transformers import glue_output_modes as output_modes
#from transformers import glue_processors as processors
from processor_ddie import ddie_processors as processors
from transformers import glue_convert_examples_to_features as convert_examples_to_features


logger = logging.getLogger(__name__)

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, XLNetConfig, XLMConfig, 
                                                                                RobertaConfig, DistilBertConfig)), ())

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer)
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


#def train(args, train_dataset, model, tokenizer, storage_model):
def train(args, train_dataset, model, tokenizer, desc_tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    #optimizer = RAdam(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)
    if args.parameter_averaging:
        storage_model = copy.deepcopy(model)
        storage_model.zero_init_params()
    else:
        storage_model = None

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    # fingerprint = np.load(os.path.join(args.fingerprint_dir, 'corpus_train.npy'), allow_pickle=True)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    #for _ in train_iterator:
    for epoch, _ in enumerate(train_iterator, start=1):
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            # fp_indices = batch[11]
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'relative_dist1': batch[3],
                      'relative_dist2': batch[4],
                    #   'desc1_ii':       batch[5],
                    #   'desc1_am':       batch[6],
                    #   'desc1_tti':      batch[7],
                    #   'desc2_ii':       batch[8],
                    #   'desc2_am':       batch[9],
                    #   'desc2_tti':      batch[10],
                    #   'fingerprint':    fingerprint[fp_indices],
                      'labels':         batch[5],}
            if args.model_type != 'distilbert':
                inputs['token_type_ids'] = batch[2] if args.model_type in ['bert', 'xlnet'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0 and not args.tpu:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                if not args.parameter_averaging:
                    scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        #results = evaluate(args, model, tokenizer)
                        results = evaluate(args, model, tokenizer, desc_tokenizer)
                        for key, value in results.items():
                            tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss)/args.logging_steps, global_step)
                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)

            if args.tpu:
                args.xla_model.optimizer_step(optimizer, barrier=True)
                model.zero_grad()
                global_step += 1

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

            if args.parameter_averaging:
                storage_model.accumulate_params(model)

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

        if args.evaluate_during_training:
            prefix = 'epoch' + str(epoch)
            output_dir = os.path.join(args.output_dir, prefix)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            if args.parameter_averaging:
                storage_model.average_params()
                result = evaluate(args, storage_model, tokenizer, desc_tokenizer, prefix=prefix)
                storage_model.restore_params()
            else:
                results = evaluate(args, model, tokenizer, desc_tokenizer, prefix=prefix)

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    #return global_step, tr_loss / global_step
    return global_step, tr_loss / global_step, storage_model


#def evaluate(args, model, tokenizer, prefix=""):
def evaluate(args, model, tokenizer, desc_tokenizer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + '-MM') if args.task_name == "mnli" else (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        #eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, desc_tokenizer, evaluate=True, data_type="test")
        # eval_dataset = torch.load("ddi_test.pt")

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # fingerprint = np.load(os.path.join(args.fingerprint_dir, 'corpus_dev.npy'), allow_pickle=True)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            # fp_indices = batch[11]
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids':      batch[0],
                          'attention_mask': batch[1],
                          'relative_dist1': batch[3],
                          'relative_dist2': batch[4],
                        #   'desc1_ii':       batch[5],
                        #   'desc1_am':       batch[6],
                        #   'desc1_tti':      batch[7],
                        #   'desc2_ii':       batch[8],
                        #   'desc2_am':       batch[9],
                        #   'desc2_tti':      batch[10],
                        #   'fingerprint':    fingerprint[fp_indices],
                          'labels':         batch[5],}
                if args.model_type != 'distilbert':
                    inputs['token_type_ids'] = batch[2] if args.model_type in ['bert', 'xlnet'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)
        try:
            np.save(os.path.join(args.output_dir, 'preds'), preds)
            np.save(os.path.join(args.output_dir, 'labels'), out_label_ids)
        except:
            print("Np.save is brokening... No problem")
        eval_loss = eval_loss / nb_eval_steps
        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)
        result = compute_metrics(eval_task, preds, out_label_ids)
        results.update(result)

        output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {} *****".format(prefix))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    return results


#def load_and_cache_examples(args, task, tokenizer, evaluate=False):
def load_and_cache_examples(args, task, tokenizer, desc_tokenizer, evaluate=False, data_type='no'):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}'.format(
        'dev' if evaluate else 'train',
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        str(task)))
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        # label_list = processor.get_labels()
        label_list = ['false', 'mechanism', 'effect', 'advise', 'int']
        if task in ['mnli', 'mnli-mm'] and args.model_type in ['roberta']:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1] 
        if data_type=="train":
            examples = torch.load("examples_train.pt")
            for e_idx in range(len(examples)):
                if examples[e_idx].label == 'negative':
                    examples[e_idx].label = 'false'
        elif data_type=="test":
            examples = torch.load("examples_test.pt")
            for e_idx in range(len(examples)):
                if examples[e_idx].label == 'negative':
                    examples[e_idx].label = 'false'
        else:
            examples = processor.get_dev_examples(args.data_dir) if evaluate else processor.get_train_examples(args.data_dir)

        features = convert_examples_to_features(examples,
                                                tokenizer,
                                                label_list=label_list,
                                                max_length=args.max_seq_length,
                                                output_mode=output_mode,
                                                pad_on_left=bool(args.model_type in ['xlnet']),                 # pad on the left for xlnet
                                                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
        )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    # Drug Description
    desc_max_seq_length = args.desc_max_seq_length
    desc_processor = processors['desc']()
    output_mode = output_modes[task]
    # For Drug1
    # Load data features from cache or dataset file

    # all_desc_features = []
    # for drug_indx in (1,2):
    #     cached_desc_features_file = os.path.join(args.data_dir, 'cached_desc{}_{}_{}_{}_{}'.format(
    #         drug_indx,
    #         'dev' if evaluate else 'train',
    #         list(filter(None, args.model_name_or_path.split('/'))).pop(),
    #         str(desc_max_seq_length),
    #         str(task)))
    #     if os.path.exists(cached_desc_features_file) and not args.overwrite_cache:
    #         logger.info("Loading description of drug%s features from cached file %s", drug_indx, cached_desc_features_file)
    #         desc_features = torch.load(cached_desc_features_file)
    #     else:
    #         logger.info("Creating description of drug%s features from dataset file at %s", drug_indx, args.data_dir)
    #         # label_list = desc_processor.get_labels()
    #         label_list = ['negative', 'mechanism', 'effect', 'advise', 'int']
    #         if task in ['mnli', 'mnli-mm'] and args.model_type in ['roberta']:
    #             # HACK(label indices are swapped in RoBERTa pretrained model)
    #             label_list[1], label_list[2] = label_list[2], label_list[1] 
    #         desc_examples = desc_processor.get_dev_examples(args.data_dir, drug_indx) if evaluate else desc_processor.get_train_examples(args.data_dir, drug_indx)
    #         desc_features = convert_examples_to_features(desc_examples,
    #                                                 desc_tokenizer,
    #                                                 label_list=label_list,
    #                                                 max_length=desc_max_seq_length,
    #                                                 output_mode=output_mode,
    #                                                 pad_on_left=bool(args.model_type in ['xlnet']),                 # pad on the left for xlnet
    #                                                 pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
    #                                                 pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
    #         )
    #         if args.local_rank in [-1, 0]:
    #             logger.info("Saving description of drug%s features into cached file %s", drug_indx, cached_desc_features_file)
    #             torch.save(desc_features, cached_desc_features_file)
    #     all_desc_features.append(desc_features)


    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Get Position index
    drug_id = tokenizer.vocab['drug']
    one_id = tokenizer.vocab['##1']
    two_id = tokenizer.vocab['##2']

    all_input_ids = [f.input_ids for f in features]
    # print(len(all_input_ids))
    # print(all_input_ids[0].shape)
    all_entity1_pos= []
    all_entity2_pos= []
    for input_ids in all_input_ids:
        entity1_pos = args.max_seq_length-1 
        entity2_pos = args.max_seq_length-1 
        for i in range(args.max_seq_length):
            if input_ids[i] == drug_id and input_ids[i+1] == one_id:
                entity1_pos = i
            if input_ids[i] == drug_id and input_ids[i+1] == two_id:
                entity2_pos = i
        all_entity1_pos.append(entity1_pos)
        all_entity2_pos.append(entity2_pos)
    assert len(all_input_ids) == len(all_entity1_pos) == len(all_entity2_pos)

    range_list = list(range(args.max_seq_length, 2*args.max_seq_length))
    all_relative_dist1 = torch.tensor([[x - e1 for x in range_list] for e1 in all_entity1_pos], dtype=torch.long)
    all_relative_dist2 = torch.tensor([[x - e2 for x in range_list] for e2 in all_entity2_pos], dtype=torch.long)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

    # all_desc1_ii = torch.tensor([f.input_ids for f in all_desc_features[0]], dtype=torch.long)
    # all_desc1_am = torch.tensor([f.attention_mask for f in all_desc_features[0]], dtype=torch.long)
    # all_desc1_tti = torch.tensor([f.token_type_ids for f in all_desc_features[0]], dtype=torch.long)
    # all_desc2_ii = torch.tensor([f.input_ids for f in all_desc_features[1]], dtype=torch.long)
    # all_desc2_am = torch.tensor([f.attention_mask for f in all_desc_features[1]], dtype=torch.long)
    # all_desc2_tti = torch.tensor([f.token_type_ids for f in all_desc_features[1]], dtype=torch.long)

    all_desc1_ii = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_desc1_am = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_desc1_tti = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_desc2_ii = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_desc2_am = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_desc2_tti = torch.tensor([f.input_ids for f in features], dtype=torch.long)

    # Fingerprint
    fingerprint_indices = torch.tensor(list(range(len(features))), dtype=torch.long)

    #dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    #dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_relative_dist1, all_relative_dist2, all_labels)
    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids,
                            all_relative_dist1, all_relative_dist2,
                            all_desc1_ii, all_desc1_am, all_desc1_tti,
                            all_desc2_ii, all_desc2_am, all_desc2_tti,
                            fingerprint_indices,
                            all_labels)
    return dataset

class GNN_Config:
    def __init__(self, preprocess_config_json, dim, layer_hidden, layer_output, mode, activation):
        with open(preprocess_config_json, 'r') as f:
            preprocess_config = json.load(f)
        self.N_fingerprints = preprocess_config['N_fingerprints']
        self.radius = preprocess_config['radius']
        self.dim = dim
        self.layer_hidden = layer_hidden
        self.layer_output = layer_output
        self.mode = mode
        self.activation = activation

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=10000,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=10000,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--tpu', action='store_true',
                        help="Whether to run on the TPU defined in the environment variables")
    parser.add_argument('--tpu_ip_address', type=str, default='',
                        help="TPU IP address if none are set in the environment variables")
    parser.add_argument('--tpu_name', type=str, default='',
                        help="TPU name if none are set in the environment variables")
    parser.add_argument('--xrt_tpu_config', type=str, default='',
                        help="XRT TPU config if none are set in the environment variables")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")

    parser.add_argument('--parameter_averaging', action='store_true', help="Whether to use parameter averaging")
    parser.add_argument("--dropout_prob", default=.0, type=float, help="Dropout probability")
    parser.add_argument('--middle_layer_size', type=int, default=0, help="Dimention of middle layer")

    ## For CNN
    parser.add_argument('--use_cnn', action='store_true', help="Whether to use CNN")
    parser.add_argument('--conv_window_size', type=int, nargs='+', default=[3], help="List of convolution window size")
    #parser.add_argument('--conv_window_size', type=int, help="Convolution window size")
    parser.add_argument('--pos_emb_dim', type=int, default=10, help="Dimention of position embeddings.")
    parser.add_argument('--activation', type=str, default='relu', help="Activation function")
    ## For Drug Description
    parser.add_argument('--use_desc', action='store_true', help="Whether to use drug description")
    parser.add_argument('--desc_max_seq_length', type=int, default=128, help="Window size of convolution for desc")
    parser.add_argument('--desc_conv_window_size', type=int, help="Window size of convolution for desc")
    parser.add_argument('--desc_conv_output_size', type=int, help="Output size of convolution for desc")
    parser.add_argument('--desc_layer_hidden', type=int, default=0, help="The number of hidden layer")
    ## For Molecular Structure
    parser.add_argument('--use_mol', action='store_true', help="Whether to use molecular structure information")
    parser.add_argument('--fingerprint_dir', type=str, help="The path to fingerprint .npy files")
    parser.add_argument('--molecular_vector_size', type=int, help="Dimention of molecular embeddings.")
    parser.add_argument('--gnn_layer_hidden', type=int, help="The number of hidden layer")
    parser.add_argument('--gnn_layer_output', type=int, help="The number of output layer")
    parser.add_argument('--gnn_mode', type=str, help="The method of aggregating atom vectors")
    parser.add_argument('--gnn_activation', type=str, default='relu', help="GNN activation function")

    parser.add_argument('--pretrained_dir', type=str, help="The path to pre-trained model dir")
    parser.add_argument('--pretrained_gnn_dir', type=str, default=None, help="The path to pre-trained GNN model dir")
    parser.add_argument('--pretrained_desc_dir', type=str, default=None, help="The path to pre-trained desc BERT model dir")
    parser.add_argument('--freeze_pretrained_parameters', action='store_true', help="Whether to freeze parameters pretrained on database")

    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    if args.tpu:
        if args.tpu_ip_address:
            os.environ["TPU_IP_ADDRESS"] = args.tpu_ip_address
        if args.tpu_name:
            os.environ["TPU_NAME"] = args.tpu_name
        if args.xrt_tpu_config:
            os.environ["XRT_TPU_CONFIG"] = args.xrt_tpu_config

        assert "TPU_IP_ADDRESS" in os.environ
        assert "TPU_NAME" in os.environ
        assert "XRT_TPU_CONFIG" in os.environ

        import torch_xla
        import torch_xla.core.xla_model as xm
        args.device = xm.xla_device()
        args.xla_model = xm

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path, num_labels=num_labels, finetuning_task=args.task_name)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)
    desc_tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)
    #model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path), config=config)
    # gnn_config = GNN_Config(os.path.join(args.fingerprint_dir, 'config.json'), args.molecular_vector_size, args.gnn_layer_hidden, args.gnn_layer_output, args.gnn_mode, args.gnn_activation)
    model = model_class(args, config, None)
    if args.use_mol and args.pretrained_gnn_dir is not None:
        model.gnn.load_state_dict(torch.load(os.path.join(args.pretrained_gnn_dir, 'gnn_state_dict')))
        if args.freeze_pretrained_parameters:
            for param in model.gnn.parameters():
                param.requires_grad = False
    if args.use_desc and args.pretrained_desc_dir is not None:
        model.desc_bert.load_state_dict(torch.load(os.path.join(args.pretrained_desc_dir, 'desc_bert_state_dict')))
        model.desc_conv.load_state_dict(torch.load(os.path.join(args.pretrained_desc_dir, 'desc_conv_state_dict')))
        if args.freeze_pretrained_parameters:
            for param in model.desc_bert.parameters():
                param.requires_grad = False
            for param in model.desc_conv.parameters():
                param.requires_grad = False

    if not args.do_train:
        global_step = 0
        model.load_state_dict(torch.load(os.path.join(args.pretrained_dir, 'state_dict')))
    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        #train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, desc_tokenizer, evaluate=False, data_type="train")
        # train_dataset =torch.load ("ddi_train.pt")
        global_step, tr_loss, storage_model =  train(args, train_dataset, model, tokenizer, desc_tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0) and not args.tpu:
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        torch.save(model.state_dict(), os.path.join(args.output_dir, 'state_dict'))
        '''Do not save model
        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        
        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)

        # Save tokenizer
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir, config, gnn_config)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        model.to(args.device)
        '''


    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        #tokenizer = tokenizer_class.from_pretrained(args.finetuned_dir, do_lower_case=args.do_lower_case)
        if args.parameter_averaging:
            #storage_model.load_state_dict(torch.load(os.path.join('/mnt/output/foo', 'state_dict_epoch5')))
            storage_model.average_params()
            result = evaluate(args, storage_model, tokenizer, desc_tokenizer, prefix="")
        else:
            #result = evaluate(args, model, tokenizer, prefix="")
            result = evaluate(args, model, tokenizer, desc_tokenizer, prefix="")
        result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
        results.update(result)

        '''Do not use checkpoints
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split('/')[-1] if checkpoint.find('checkpoint') != -1 else ""
            
            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, model, tokenizer, prefix=prefix)
            result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
            results.update(result)
        '''
    return results

if __name__ == "__main__":
    main()
