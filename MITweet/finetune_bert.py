from bert_model import BertForMultiClassification
from config import *
from transformers import AutoModel, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import ast
import argparse


def main(**kwargs):
    base_model = AutoModel.from_pretrained(base_model_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    split = load_dataset("csv", data_files=data_files)

    frozen_base = False
    if "frozen_base" in kwargs:
        frozen_base = kwargs["frozen_base"]

    print("Frozen base:", frozen_base)

    def tokenizer_preprocess(ds):
        ds["labels"] = [ast.literal_eval(item) for item in ds["labels"]]

        return tokenizer(
            ds["tweet"],
            truncation=True,
            padding='max_length',
            max_length=max_length
        )

    split = split.map(tokenizer_preprocess, remove_columns=["topic", "tweet"], batched=True)

    model = BertForMultiClassification(
        base_model=base_model,
        num_dims=num_dims,
        num_labels_per_dim=num_labels_per_dim,
        frozen_base=frozen_base
    )

    save_dir_prefix = "bert_frozenbase" if frozen_base else "bert_fullparam"

    train_args = TrainingArguments(
        **train_config,
        output_dir=f"{save_dir_prefix}_checkpoints",
        logging_dir=f"{save_dir_prefix}_training_logs"
    )

    def compute_metrics(eval_pred: tuple[np.ndarray]):
        logits, labels = eval_pred  
        preds = np.argmax(logits, axis=-1)  # Shape like: [N, num_dim]

        per_dim_acc = []
        per_dim_micro_f1 = []
        per_dim_macro_f1 = []

        for dim in range(num_dims):
            true_dim = labels[:, dim]
            pred_dim = preds[:, dim]
            
            acc = accuracy_score(true_dim, pred_dim)
            micro_f1 = f1_score(true_dim, pred_dim, average='micro')
            macro_f1 = f1_score(true_dim, pred_dim, average='macro')

            per_dim_acc.append(acc)
            per_dim_micro_f1.append(micro_f1)
            per_dim_macro_f1.append(macro_f1)

        mean_acc = np.mean(per_dim_acc)
        mean_micro_f1 = np.mean(per_dim_micro_f1)
        mean_macro_f1 = np.mean(per_dim_macro_f1)

        return {
            'mean_acc': mean_acc,
            'mean_micro_f1': mean_micro_f1,
            'mean_macro_f1': mean_macro_f1
        }

    print('First data:')
    print(split['train'][0])

    model.train()
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=split['train'],
        eval_dataset=split['eval'],
        compute_metrics=compute_metrics
    )

    ret = trainer.train()

    print(ret)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--frozen_base', type=bool, default=False)

    args = parser.parse_args()

    main(
        frozen_base=args.frozen_base
    )