
base_model_path = "/home/ccm/models/bert-base-uncased"

train_ds_path = "processed_data/train.csv"
eval_ds_path = "processed_data/eval.csv"
test_ds_path = "processed_data/test.csv"

data_files = {
    "train": train_ds_path,
    "eval": eval_ds_path,
    "test": test_ds_path
}


label_columns = ["I1", "I2", "I3", "I4", "I5", "I6", "I7", "I8", "I9", "I10", "I11", "I12"]

num_dims = len(label_columns)
num_labels_per_dim = 4
max_length = 256

train_config = {
    # train_process_config
    "seed": 42,
    "num_train_epochs": 10,
    # output_config
    "overwrite_output_dir": True,
    "save_strategy": "epoch",  # Save model checkpoint at each epoch
    "save_total_limit": 10,
    # log_config
    "report_to": ['tensorboard'],
    "logging_strategy": "epoch",  # Log model state at each epoch
    # evaluation_config
    "eval_strategy": "epoch",
    "load_best_model_at_end": True,
    "metric_for_best_model": "mean_acc",  
    # dataloader_config
    "dataloader_num_workers": 4,
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 16,
    # optimizer_config (Default optimizer: AdamW)
    "learning_rate": 5e-5,
    "weight_decay": 0.1,
    # learning_rate scheduler 
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.05  
}