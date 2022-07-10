

import time
import opendelta
import argparse
import os
import jsonlines
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from openprompt.data_utils import InputExample
from openprompt.plms import load_plm
from tqdm import tqdm
import torch

import re
import numpy as np
from collections import Counter
import string


parser = argparse.ArgumentParser("")
parser.add_argument("--data_path", type=str, default="./data/ARC-DA-v1.1")
parser.add_argument("--plm_path", type=str, default="./plm_cache/opt-350m")
parser.add_argument("--delta_type", type=str, default="adapter")
parser.add_argument("--modified_modules", type=str,
                    nargs="+", default=["self_attn", "fc2"])
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--max_seq_l", type=int, default=128)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--delta_save_path", default="./delta_ckpts/")
parser.add_argument("--raw_delta_save_path", default="./raw_delta_ckpts/")
parser.add_argument("--train_epochs", type=int, default=3)
parser.add_argument("--warmup_steps", type=int, default=0)
parser.add_argument("--mode", type=str, default="train",
                    choices=["train", "interactive_inference"])
parser.add_argument("--push_to_dc", type=bool, default=False)

args = parser.parse_args()


def get_dataset(args):
    #############################################################
    # 1. load the dataset (three splits: train, dev, test) here from the jsonl file
    # 2. transform each data sample into an InputExample with can be consumed by openprompt.
    # 3. return the train, dev, test dataset in the form of a dict, where each value is a list of InputExample(s).
    # Hint1: the dataset provide multiple answers for one question. To keep the pipeline simple, you can use the first answer as the target answer.
    # Hint2: the label of InputExample should be set to int(0) to match the verbalizer.
    #############################################################
    dataset = {}
    for split in ['train', 'dev', 'test']:
        dataset[split] = []
        with jsonlines.open(os.path.join(args.data_path, split + '.jsonl')) as reader:
            for data in reader:
                input_example = InputExample(
                    text_a=data['question'], label=0, meta={'answer': data['answers'][0]})
                dataset[split].append(input_example)
    return dataset


def add_delta_to_backbone(plm, args):
    #############################################################
    # 1. add delta model to the backbone
    # Hint: using AutoDeltaConfig.from_dict, and AutoDeltaModel.from_config method
    # 2. freeze the backbone model
    # 3. visualize the modified structure to check if you've done correctly.
    #############################################################
    from opendelta import AdapterModel
    delta_model = AdapterModel(
        backbone_model=plm, modified_modules=args.modified_modules, bottleneck_dim=12)
    delta_model.freeze_module()
    delta_model.log()
    return delta_model


def load_from_finetuned(plm, args):
    #############################################################
    # 1. this is used in interactive inference
    # 2. use AutoDeltaModel.from_finetuned method to load the delta model you've just trained.
    #############################################################
    from opendelta import AutoDeltaModel
    delta_model = AutoDeltaModel.from_finetuned(args.delta_save_path, plm)
    return delta_model


def get_prompt_template(tokenizer, args):
    from openprompt.prompts import ManualTemplate
    #############################################################
    # Use the ManualTemplate to define a prompt template
    #############################################################
    mytemplate = ManualTemplate(
        text='{"placeholder":"text_a"} {"mask"}',
        tokenizer=tokenizer,
    )
    return mytemplate


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def qa_f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def get_token_f1(predictions, groud_truths):
    f1s = []
    for (prediction, dp) in zip(predictions, groud_truths):
        f1s.append(qa_f1_score(prediction, dp))
    return np.mean(f1s)


def evaluate(prompt_model, dataloader, args):
    generation_arguments = {
        "max_length": 16 + args.max_seq_l,
    }
    predictions = []
    ground_truths = []
    # print("begin evaluation")

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluation")
        for step, inputs in enumerate(pbar):
            inputs = inputs.cuda()
            _, output_sentence = prompt_model.generate(
                inputs, **generation_arguments, verbose=False)
            predictions.extend(output_sentence)
            ground_truths.extend(inputs['tgt_text'])
        assert len(predictions) == len(
            ground_truths), (len(predictions), len(ground_truths))
        # print(f"predictions {predictions[0]}, ground_truths {ground_truths[0]}")
        predictions = [prediction.strip() for prediction in predictions]
        ground_truths = [ground_truth.strip()
                         for ground_truth in ground_truths]
        # shown one example

        score = get_token_f1(predictions, ground_truths)
    return score


def main(args):
    dataset = get_dataset(args)
    plm, tokenizer, model_config, WrapperClass = load_plm(
        "opt", f"{args.plm_path}")

    delta_model = add_delta_to_backbone(plm, args)

    from openprompt import PromptForGeneration
    mytemplate = get_prompt_template(tokenizer, args)

    from openprompt.prompts import GenerationVerbalizer
    from openprompt import PromptDataLoader
    myverbalizer = GenerationVerbalizer(tokenizer, classes=None, label_words={
                                        0: ["{'meta':'answer'}"]}, is_rule=True)

    prompt_model = PromptForGeneration(plm=plm, template=mytemplate)

    prompt_model.cuda()

    train_dataloader = PromptDataLoader(dataset=dataset["train"], template=mytemplate, verbalizer=myverbalizer, tokenizer=tokenizer,
                                        tokenizer_wrapper_class=WrapperClass, max_seq_length=args.max_seq_l,
                                        batch_size=args.batch_size, shuffle=True, teacher_forcing=True, predict_eos_token=True,
                                        truncate_method="tail")

    validation_dataloader = PromptDataLoader(dataset=dataset["dev"], template=mytemplate, verbalizer=myverbalizer, tokenizer=tokenizer,
                                             tokenizer_wrapper_class=WrapperClass, max_seq_length=args.max_seq_l,
                                             batch_size=args.batch_size, shuffle=False, teacher_forcing=False, predict_eos_token=False,
                                             truncate_method="tail")

    test_dataloader = PromptDataLoader(dataset=dataset["test"], template=mytemplate, verbalizer=myverbalizer, tokenizer=tokenizer,
                                       tokenizer_wrapper_class=WrapperClass, max_seq_length=args.max_seq_l,
                                       batch_size=args.batch_size, shuffle=False, teacher_forcing=False, predict_eos_token=False,
                                       truncate_method="tail")

    ######################################################################
    # Define the optimizer and scheduler using normal pytorch ways.
    #####################################################################
    optimizer = AdamW(prompt_model.parameters(), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.train_epochs)

    best_var_f1, tot_loss = 0, 0

    print('-' * 89)
    for epoch in range(args.train_epochs):
        pbar = tqdm(train_dataloader, desc=f"Epoch{epoch}")
        epoch_start = time.time()
        start_time = time.time()
        for step, inputs in enumerate(pbar):

            #############################################################
            # Complete the training loop using pytorch style
            #############################################################

            inputs = inputs.cuda()
            loss = prompt_model(inputs)
            loss.backward()
            tot_loss += loss.item()
            optimizer.step()
            prompt_model.zero_grad()
            optimizer.zero_grad()
            args.log_interval = 100
            if (step % args.log_interval == 0 and step > 0) or step == len(pbar) - 1:
                elapsed = time.time() - start_time
                if (step % args.log_interval == 0):
                    cur_loss = tot_loss / args.log_interval
                    elapsed = elapsed * 1000 / args.log_interval
                else:
                    cur_loss = tot_loss / (step % args.log_interval)
                    elapsed = elapsed * 1000 / (step % args.log_interval)
                print('| epoch {:3d} | {:5d}/{:5d} batches | '
                      'lr {:02.4f} | ms/batch {:5.2f} | '
                      'loss {:5.4f} |'.format(
                          epoch, step + 1, len(pbar),
                          optimizer.param_groups[0]['lr'],
                          elapsed, cur_loss))
                tot_loss = 0
                start_time = time.time()
            pbar.set_postfix(loss=loss.item())

        val_f1 = evaluate(prompt_model, validation_dataloader,
                          args)  # is slow due to generaiton
        scheduler.step()
        # print(f"ValAcc {val_f1}")
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid acc {:5.2%} | '.format(epoch, (time.time() - epoch_start),
                                                                                    val_f1))

        if val_f1 >= best_var_f1:
            # temperarily save the model ckpt
            if not os.path.exists(f"{args.raw_delta_save_path}"):
                os.mkdir(f"{args.raw_delta_save_path}")
            delta_model.save_checkpoint(f"{args.raw_delta_save_path}")
            print('| Model saved to {}'.format(args.raw_delta_save_path))
            best_var_f1 = val_f1
        print('-' * 89)

    # load the best model
    delta_model.load_checkpoint(f"{args.raw_delta_save_path}")

    test_f1 = evaluate(prompt_model, test_dataloader, args)
    print(f"TestAcc {test_f1}")

    if args.push_to_dc:
        delta_model.save_finetuned(f"{args.delta_save_path}/",
                                   push_to_dc=True,
                                   center_args={"name": f"{args.plm_path.split('/')[-1]}-{args.delta_type}-for-direct-qa",
                                                "test_performance": test_f1,
                                                "test_metric": "token_f1",
                                                "delta_type": args.delta_type,
                                                "train_dataset": "ARC-DA-v1.1",
                                                },
                                   center_args_pool={**vars(args)},
                                   list_tags=['Direct QA'],
                                   dict_tags={
                                       "author": "ENTER_YOUR_NAME_HERE"},
                                   delay_push=True,
                                   )
    else:
        delta_model.save_finetuned(f"{args.delta_save_path}/",
                                   push_to_dc=False,
                                   )


def get_generated_sequence(prompt_model, dataloader, args):
    generation_arguments = {
        "max_length": 16 + args.max_seq_l,
    }
    predictions = []
    print("begin evaluation")

    for step, inputs in enumerate(dataloader):
        inputs = inputs.cuda()
        _, output_sentence = prompt_model.generate(
            inputs, **generation_arguments, verbose=False)
        predictions.extend(output_sentence)

    print("Answer:", predictions[0])


def interactive_inference(args):
    plm, tokenizer, model_config, WrapperClass = load_plm(
        "opt", f"{args.plm_path}")
    delta_model = load_from_finetuned(plm, args)
    from openprompt import PromptForGeneration
    mytemplate = get_prompt_template(tokenizer, args)

    from openprompt import PromptDataLoader
    prompt_model = PromptForGeneration(plm=plm, template=mytemplate)
    prompt_model.cuda()

    while True:
        question = input("Input a question:")
        if question.strip() == "quit()":
            break
        inference_dataset = [InputExample(text_a=question, label=0)]
        inference_dataloader = PromptDataLoader(dataset=inference_dataset, template=mytemplate, verbalizer=None, tokenizer=tokenizer,
                                                tokenizer_wrapper_class=WrapperClass, max_seq_length=args.max_seq_l,
                                                batch_size=args.batch_size, shuffle=False, teacher_forcing=False, predict_eos_token=False,
                                                truncate_method="tail")
        get_generated_sequence(prompt_model, inference_dataloader, args)


if __name__ == "__main__":
    if args.mode == "train":
        main(args)
    elif args.mode == "interactive_inference":
        interactive_inference(args)
