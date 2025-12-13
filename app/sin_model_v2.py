# app/sin_model_v2.py

import torch
import torch.nn as nn
from transformers import AutoModel

class SinhalaRegressorV2(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()

        # Load encoder ONLY ONCE
        self.encoder = AutoModel.from_pretrained(model_name)

        hidden_size = self.encoder.config.hidden_size

        # Multi-head regression heads
        self.richness = nn.Linear(hidden_size, 1)
        self.organization = nn.Linear(hidden_size, 1)
        self.technical = nn.Linear(hidden_size, 1)
        self.total = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        cls = outputs.last_hidden_state[:, 0]

        return {
            "richness_5": self.richness(cls),
            "organization_6": self.organization(cls),
            "technical_3": self.technical(cls),
            "total_14": self.total(cls),
        }
