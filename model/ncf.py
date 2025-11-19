import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

class MLPBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dropout: float
    ):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x

class NCF(nn.Module):
    def __init__(
        self,
        user_num: int,
        item_num: int,
        embed_dim: int,
        dropout: float
    ):
        super().__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.embed_dim = embed_dim
        self.dropout = dropout

        self.user_embedding_mf = nn.Embedding(user_num, embed_dim)
        self.item_embedding_mf = nn.Embedding(item_num, embed_dim)

        self.user_embedding_mlp = nn.Embedding(user_num, embed_dim)
        self.item_embedding_mlp = nn.Embedding(item_num, embed_dim)

        mlp_input_dim = embed_dim * 2
        self.mlp_block1 = MLPBlock(mlp_input_dim, 128, dropout)
        self.mlp_block2 = MLPBlock(128, 64, dropout)
        self.mlp_block3 = MLPBlock(64, 32, dropout)

        self.output_layer = nn.Linear(embed_dim + 32, 1)

    def forward(self, user_indices, item_indices):
        user_embed_mf = self.user_embedding_mf(user_indices)
        item_embed_mf = self.item_embedding_mf(item_indices)
        mf_vector = user_embed_mf * item_embed_mf

        user_embed_mlp = self.user_embedding_mlp(user_indices)
        item_embed_mlp = self.item_embedding_mlp(item_indices)
        mlp_vector = torch.cat([user_embed_mlp, item_embed_mlp], dim=-1)
        mlp_vector = self.mlp_block1(mlp_vector)
        mlp_vector = self.mlp_block2(mlp_vector)
        mlp_vector = self.mlp_block3(mlp_vector)

        final_vector = torch.cat([mf_vector, mlp_vector], dim=-1)
        logits = self.output_layer(final_vector)
        output = torch.sigmoid(logits).squeeze()

        return output
    
def evaluator(
    model,
    data_loader,
    k: int, 
    device,
    num_neg: int = 100,
    user_sample_size: int = 10000,
    is_test: bool = True
):
    user_num = data_loader.user_num
    item_num = data_loader.item_num

    train_user_pos = data_loader.train_user_pos
    if is_test:
        eval_user_pos = data_loader.test_user_pos
    else:
        eval_user_pos = data_loader.val_user_pos

    all_users = list(eval_user_pos.keys())

    if len(all_users) > user_sample_size:
        users = np.random.choice(all_users, size=user_sample_size, replace=False)
    else:
        users = all_users

    NDCG = 0.0
    HT = 0.0
    
    model.eval()
    with torch.no_grad():
        for u in users:
            eval_items = eval_user_pos[u]