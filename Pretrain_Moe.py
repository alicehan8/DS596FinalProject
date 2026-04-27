import torch
import torch.nn as nn
from utils import *
from einops.layers.torch import Rearrange
from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers import AutoConfig, AutoModelForMaskedLM



class Pretrain_Moe(nn.Module):
    def __init__(self, task_dict,freeze_layer = False,return_hidden = False):
        super(Pretrain_Moe, self).__init__()
        config_overrides = {}
        config = AutoConfig.from_pretrained(
            "kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16",
            **config_overrides,trust_remote_code=True
        )
        self.model = AutoModelForMaskedLM.from_config(config,trust_remote_code=True)
        if freeze_layer:
            for param in self.model.parameters():
                param.requires_grad = False

        self.return_hidden = return_hidden

        self.cnn = nn.Sequential(ConvBlock(16, 64, 5),  # (N,C,L)
                                 nn.MaxPool1d(2),
                                 ConvBlock(64, 128, 5),
                                 nn.MaxPool1d(2),
                                 ConvBlock(128, 256, 5),
                                 nn.MaxPool1d(2),
                                 ConvBlock(256, 128, 5),
                                 nn.MaxPool1d(2),
                                )

        self.blocks = TransformerMoETaskGating(task_dict = task_dict,embed_dim=128, depth=2, num_heads=4,return_hidden = self.return_hidden)

    def forward(self, input_ids):
        outputs = self.model(input_ids)
        outputs = outputs['logits']

        outputs = outputs.permute(0, 2, 1)
        outputs = self.cnn(outputs)
        outputs = outputs.permute(0, 2, 1)

        if self.return_hidden:
            hidden, outputs, loss = self.blocks(outputs)
            return hidden, outputs, loss
        else:
            outputs, loss = self.blocks(outputs)
            return outputs, loss
            # return outputs['task1']

def ConvBlock(dim, dim_out = None, kernel_size =5):
    
    return nn.Sequential(nn.Conv1d(dim, dim_out, kernel_size, padding="same"),
                     # nn.BatchNorm1d(dim_out),
                     nn.ReLU(),
)

class CNN_Moe(nn.Module):
    def __init__(self, task_dict,freeze_layer = False,return_hidden = False):
        super(CNN_Moe, self).__init__()

        self.return_hidden = return_hidden

        self.cnn = nn.Sequential(ConvBlock(4, 64, 5),  # (N,C,L)
                                 nn.MaxPool1d(2),
                                 ConvBlock(64, 128, 5),
                                 nn.MaxPool1d(2),
                                 ConvBlock(128, 256, 5),
                                 nn.MaxPool1d(2),
                                 ConvBlock(256, 128, 5),
                                 nn.MaxPool1d(2),
                                )

        self.blocks = TransformerMoETaskGating(task_dict = task_dict,embed_dim=128, depth=2, num_heads=4,return_hidden = self.return_hidden)

    def forward(self, input_ids):
        # outputs = input_ids.permute(0, 2, 1)
        outputs = self.cnn(input_ids)
        outputs = outputs.permute(0, 2, 1)

        if self.return_hidden:
            hidden, outputs, loss = self.blocks(outputs)
            return hidden, outputs, loss
        else:
            outputs, loss = self.blocks(outputs)
            return outputs, loss
