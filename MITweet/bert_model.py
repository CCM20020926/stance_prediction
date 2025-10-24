import torch
from torch import nn


class BertForMultiClassification(nn.Module):
    def __init__(self, 
        base_model,
        num_dims,
        num_labels_per_dim,
        frozen_base=False,
        *args, **kwargs) -> None:

        super().__init__(*args, **kwargs)
        self.encoder = base_model
        self.num_dims = num_dims
        self.num_labels_per_dim = num_labels_per_dim
        self.frozen_base = frozen_base

        hidden_size = self.encoder.config.hidden_size
        self.linear = nn.Linear(hidden_size, num_dims * num_labels_per_dim)


    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        if self.frozen_base:
            with torch.no_grad():
                output = self.encoder(
                    input_ids,
                    attention_mask,
                    token_type_ids
                )
        
        else:
            output = self.encoder(
                input_ids,
                attention_mask,
                token_type_ids
            )
        
        pooler_output = output.pooler_output
        logits = self.linear(pooler_output)
        logits = logits.view(-1, self.num_dims, self.num_labels_per_dim) # Shape Like, [N, num_dims, num_labels_per_dim]

        loss = None
        if labels is not None:
            loss_func = nn.CrossEntropyLoss()

            loss = sum(
                [
                    loss_func(logits[:, i, :] , labels[:, i])
                    for i in range(self.num_dims)
                ]
            )

            loss = loss / self.num_dims
        
        return {"loss": loss, "logits": logits}


