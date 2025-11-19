import numpy as np
from collections import defaultdict

import torch
from sklearn.model_selection import train_test_split

from src.utils import getDF
from src.path import DATA_DIR

class NCFDataLoader:
    def __init__(
        self,
        fname: str,
        test_size: float,
        val_size: float,
        seed: int = 42
    ):
        self.fname = fname
        self.fpath = DATA_DIR / f"{fname}.jsonl.gz"
        self.test_size = test_size
        self.val_size = val_size
        self.seed = seed

        if not (0.0 < self.val_size < self.test_size < 1.0):
            raise ValueError(
                f"`val_size`({self.val_size}) must be > 0 and < `test_size`({self.test_size}) < 1.0"
            )
        
        self.raw_df = self._load_data(self.fpath)
        (
            self.user2id,
            self.item2id,
            self.user_num,
            self.item_num,
            self.train_df,
            self.val_df,
            self.test_df,
        ) = self._preprocess()

        self.train_user_pos = self._get_user_pos(self.train_df)
        self.val_user_pos = self._get_user_pos(self.val_df)
        self.test_user_pos = self._get_user_pos(self.test_df)

    def _load_data(self, path):
        return getDF(path).rename(
            columns={
                "user_id": "user",
                "asin": "item",
            }
        )[["user", "item"]]
    
    def _preprocess(self):
        df = self.raw_df.copy()
        df = df.drop_duplicates(subset=["user", "item"]).reset_index(drop=True)

        user2id = {u: i for i, u in enumerate(df["user"].unique())}
        item2id = {i: j for j, i in enumerate(df["item"].unique())}

        df["user_idx"] = df["user"].map(user2id)
        df["item_idx"] = df["item"].map(item2id)

        user_num = len(user2id)
        item_num = len(item2id)

        train_val_df, test_df = train_test_split(
            df[["use_idx", "item_idx"]],
            test_size=self.test_size,
            random_state=self.seed,
        )

        val_ratio = self.val_size / (1.0 - self.test_size)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_ratio,
            random_state=self.seed,
        )

        return user2id, item2id, user_num, item_num, train_df, val_df, test_df
    
    def _get_user_pos(self, df):
        user_pos = defaultdict(list)
        for _, (u, i) in df.iterrows():
            user_pos[u].append(i)
        return user_pos
    
    def get_bpr_batch(self, batch_size):
        users = []
        pos_items = []
        neg_items = []

        all_items = set(range(self.item_num))

        for _ in range(batch_size):
            # 1) randomly sample a user with at least one interaction in train
            u = np.random.choice(list(self.train_user_pos.keys()))
            i = np.random.choice(list(self.train_user_pos[u]))

            # 2) randomly sample a negative item (not in train interactions)
            while True:
                j = np.random.choice(all_items)
                if j not in self.train_user_pos[u]:
                    break

            users.append(u)
            pos_items.append(i)
            neg_items.append(j)

        return (
            torch.LongTensor(users),
            torch.LongTensor(pos_items),
            torch.LongTensor(neg_items),
        )
