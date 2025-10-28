import asyncio
from config import eval_ds_path, label_columns
from llm_agent import LLMAgentZeroShot, parse_json
from tqdm import tqdm
import pandas as pd
import os
import argparse

batch_size = 5

def write_result(log_df: pd.DataFrame, indexes, reasons, predictions, labels, path):
    assert len(reasons) == len(predictions) and len(predictions) == len(labels) 

    if len(indexes) > len(reasons):
        indexes = indexes[:len(reasons)]

    sub_df = pd.DataFrame(
        {
            "index": indexes,
            "reason": reasons,
            "prediction": predictions,
            "label": labels
        }
    )

    log_df = pd.concat([log_df, sub_df])

    log_df.to_csv(path, index=None, encoding='utf-8')

    return log_df  # 此处必须返回更新后的 log_df，否则无法更新调用者中的 log_df 对象


async def call_agent(agent, batch_texts):
    tasks = []

    for text in batch_texts:
        task = asyncio.create_task(agent.ainvoke(text))
        tasks.append(task)
        
    responses = await asyncio.gather(*tasks)
    parsed_responses = [parse_json(response) for response in responses]
    return parsed_responses


async def run_agent_test(model):
    agent = LLMAgentZeroShot(model)
    eval_ds = pd.read_csv(eval_ds_path)
    texts = eval_ds["tweet"]
    labels = eval_ds["labels"]

    # 1. Load the result log file
    logfile_path = f"results/{model.replace('/', '--')}.csv"
    if os.path.exists(logfile_path):
        log_df = pd.read_csv(logfile_path)

    else:
        log_df = pd.DataFrame({
            "index": [],
            "reason": [],
            "prediction": [],
            "label": []
        })
    
    start_index = 0 if len(log_df) ==0 else log_df["index"].iloc[-1].item() + 1
    print("Start Index:", start_index)
    
    for i in tqdm(range(start_index, len(eval_ds), batch_size)):
        indexes = [j for j in range(i, i + batch_size)]
        batch_texts = texts[i: i + batch_size]
        batch_labels = labels[i: i + batch_size]
        
        parsed_responses = await call_agent(agent, batch_texts)
        reasons = []
        predictions = []

        for item in parsed_responses:
            reason = item["reason"]
            result = item["result"]
            prediction = [result[col] for col in label_columns]

            reasons.append(reason)
            predictions.append(str(prediction))


        log_df = write_result(log_df, indexes, reasons, predictions, batch_labels, logfile_path)  # 此处必须接收子函数返回值才能重新更新 log_df



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)

    args = parser.parse_args()

    asyncio.run(run_agent_test(args.model))



    
    

