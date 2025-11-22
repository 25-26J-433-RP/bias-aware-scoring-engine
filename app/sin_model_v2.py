import torch.nn as nn
from transformers import AutoModel


class SinhalaRegressorV2(nn.Module):
    """
    XLM-R based regressor for Sinhala essay scoring (V2).
    Uses CLS embedding -> Linear layer -> single score output.
    """

    def __init__(self, model_name: str = "xlm-roberta-base"):
        super().__init__()

        # Transformer encoder (frozen or trainable depending on training script)
        self.encoder = AutoModel.from_pretrained(model_name)

        # Simple regression head: hidden_size -> 1 scalar score
        self.regressor = nn.Linear(self.encoder.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        """
        input_ids: Tensor [batch_size, seq_len]
        attention_mask: Tensor [batch_size, seq_len]
        returns: Tensor [batch_size, 1]
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # CLS token embedding
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [batch, hidden]

        # Regress to a single score
        score = self.regressor(cls_embedding)  # [batch, 1]
        return score
