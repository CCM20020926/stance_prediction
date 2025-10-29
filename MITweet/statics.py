import pandas as pd
import ast
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from config import label_columns
import json

def calculate_statics(model:str):
    # 1. Load log file
    logfile_path = f"results/{model.replace('/', '--')}.csv"
    df = pd.read_csv(logfile_path)
    predictions = df["prediction"]
    labels = df["label"]

    # 2. Deserialize
    predictions = np.array([ast.literal_eval(item) for item in predictions])
    labels = np.array([ast.literal_eval(item) for item in labels])

    # 3. Calculate mACC, mMICRO-F1, mMACRO-F1
    acc_per_dim = []
    micro_f1_per_dim = []
    macro_f1_per_dim = []

    for i in range(len(label_columns)):
        prediction_per_dim = predictions[:, i]
        label_per_dim = labels[:, i]

        acc = accuracy_score(label_per_dim, prediction_per_dim)
        micro_f1 = f1_score(label_per_dim, prediction_per_dim, average='micro')
        macro_f1 = f1_score(label_per_dim, prediction_per_dim, average='macro')

        acc_per_dim.append(acc)
        micro_f1_per_dim.append(micro_f1)
        macro_f1_per_dim.append(macro_f1)

    
    macc = np.array(acc_per_dim).mean().item()
    mmicro_f1 = np.array(micro_f1_per_dim).mean().item()
    mmacro_f1 = np.array(macro_f1_per_dim).mean().item()

    result =  {
        "mean_accuracy": macc,
        "mean_micro_f1": mmicro_f1,
        "mean_macro_f1": mmacro_f1
    }

    with open(f"results/{model.replace('/', '--')}_statics.json", 'w') as f:
        json.dump(result, f)


if __name__ == '__main__':
    models = ['gpt-5', 'deepseek/deepseek-v3', 'deepseek/deepseek-r1']

    for model in models:
        calculate_statics(model)