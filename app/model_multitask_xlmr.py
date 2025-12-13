import torch
import torch.nn as nn
from transformers import PreTrainedModel, AutoModel, AutoConfig

class SinhalaMultiHeadRegressor(PreTrainedModel):
    config_class = AutoConfig

    def __init__(self, config):
        super().__init__(config)

        self.encoder = AutoModel.from_config(config)
        hidden = config.hidden_size

        # ---- Grade embedding ----
        self.grade_embed = nn.Embedding(10, hidden)

        self.dropout = nn.Dropout(0.2)

        self.richness = nn.Linear(hidden, 1)
        self.organization = nn.Linear(hidden, 1)
        self.technical = nn.Linear(hidden, 1)
        self.total = nn.Linear(hidden, 1)

    def forward(self, input_ids, attention_mask, grade_id):
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        cls = out.last_hidden_state[:, 0, :]
        grade_vec = self.grade_embed(grade_id)

        x = self.dropout(cls + grade_vec)

        return {
            "richness_5": self.richness(x).squeeze(-1),
            "organization_6": self.organization(x).squeeze(-1),
            "technical_3": self.technical(x).squeeze(-1),
            "total_14": self.total(x).squeeze(-1),
        }
