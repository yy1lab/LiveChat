import random
import numpy as np
import torch
import json
import os
import h5py
from torch.utils.data import Dataset


class GameBased(Dataset):
    """
    Custom PyTorch Dataset class for the Twitch-FIFA dataset.

    Args:
        tokenizer (transformers.AutoTokenizer): Tokenizer to encode chat text and other inputs.
        root (str, optional): Root directory path for the dataset. Defaults to "/media/livechat/game_based_dialogue/".
        dataset_json (str, optional): JSON file containing dataset information. Defaults to "train.json".
        video_feature_file (str, optional): File containing video features in h5 format. Defaults to "train_video_feat.h5".
        comments_padding (int, optional): Maximum length of chat comments after padding. Defaults to 50.
        nb_context_comments (int, optional): Number of context comments to include in the chat context. Defaults to 30.
        mode (str, optional): Mode of operation ('train', 'eval', or 'gen'). Defaults to "train".
        allow_special_token (bool, optional): Whether to add special tokens during tokenization. Defaults to True.

    Attributes:
        root (str): Root directory path for the dataset.
        video_feature_file (str): File containing video features in h5 format.
        ds_json (list): Loaded JSON data from the dataset file.
        video_context_id (list): List of video context IDs.
        chat_context (list): List of chat contexts.
        response_indexes (list): List of response indexes.
        responses (list): List of response messages.
        h5f (h5py.File): File object for accessing video features.
        tokenizer (transformers.AutoTokenizer): Tokenizer used for encoding text sequences.
        comments_padding (int): Maximum length of chat comments after padding.
        nb_context_comments (int): Number of context comments to include in the chat context.
        allow_special_token (bool): Whether to add special tokens during tokenization.
        mode (str): Mode of operation ('train', 'eval', or 'gen').

    Methods:
        __len__(): Returns the number of samples in the dataset.
        __getitem__(index): Returns a single sample from the dataset based on the given index.

    Note:
        For evaluation mode ('eval'), the dataset must contain 'label' and 'response' fields in the JSON data.

    Example:
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        livechat = GameBased(tokenizer)
        print(len(livechat))  # Print the number of samples in the dataset.
        item = livechat[1]  # Get the second sample in the dataset.
        print(item["video_features"].size())  # Print the size of video features for the second sample.
    """
    def __init__(
            self, 
            tokenizer, 
            root="/media/livechat/game_based_dialogue/",
            dataset_json="train.json",
            video_feature_file="train_video_feat.h5",
            comments_padding=50,
            nb_context_comments=30,
            mode="train",
            allow_special_token=True
        ):
        self.root = root
        self.video_feature_file = video_feature_file
        with open(os.path.join(root, dataset_json), "r") as data:
            self.ds_json = json.load(data)
        # self.ds_json = self.ds_json[:100]

        self.video_context_id = [video["video_context_id"] for video in self.ds_json]
        self.chat_context = [video["chat_context"].split("\u0c09") for video in self.ds_json]
        self.response_indexes = [video["label"] for video in self.ds_json]
        self.responses = [video["response"] for video in self.ds_json]
        self.h5f = h5py.File(os.path.join(self.root,video_feature_file),'r')
        
        self.tokenizer = tokenizer
        self.comments_padding = comments_padding
        self.nb_context_comments = nb_context_comments
        self.allow_special_token = allow_special_token
        self.mode = mode
    
    def __len__(self):
        return len(self.video_context_id)

    def __getitem__(self, index):
        chat_context = self.chat_context[index]
        response = self.responses[index][np.argmax(self.response_indexes[index])]
        chat_context_ids = []
        chat_context_am = []
        for _ in range(self.nb_context_comments):
            random_index = random.randint(0, len(chat_context)-1)
            chat_context_input = self.tokenizer.encode_plus(chat_context[random_index], add_special_tokens=self.allow_special_token, max_length=self.comments_padding, padding='max_length', truncation=True)
            chat_context_ids.append(chat_context_input["input_ids"])
            chat_context_am.append(chat_context_input["attention_mask"])
        

        response_input = self.tokenizer.encode_plus(response, add_special_tokens=self.allow_special_token, max_length=self.comments_padding, padding='max_length', truncation=True)
        response_ids = response_input["input_ids"]
        response_am = response_input["attention_mask"]


        video_features = self.__getvideofeatures__(index)

        candidates_ids, candidates_am = self.__getcandidates__(index) if self.mode=="eval" else ([0], [0])


        return {
            "chat_context": torch.tensor(chat_context_ids, dtype=torch.long),
            "chat_context_am": torch.tensor(chat_context_am, dtype=torch.long),

            "response": torch.tensor(response_ids, dtype=torch.long),
            "response_am": torch.tensor(response_am, dtype=torch.long),

            "video_features": torch.tensor(video_features, dtype=torch.float),
                        
            "candidates": torch.tensor(candidates_ids, dtype=torch.long),
            "candidates_am": torch.tensor(candidates_am, dtype=torch.long)
        }
    
    def __getvideofeatures__(self, index):
        video_features = self.h5f[self.video_context_id[index]]
        return np.array(video_features)
    
    def __getcandidates__(self, index):
        assert self.mode=="eval"

        candidates = self.responses[index]
        candidates_ids = []
        candidates_am = []
        for candidate in candidates:
            candidate_input = self.tokenizer.encode_plus(candidate, add_special_tokens=self.allow_special_token, max_length=self.comments_padding, padding='max_length', truncation=True)
            candidates_ids.append(candidate_input["input_ids"])
            candidates_am.append(candidate_input["attention_mask"])
        return candidates_ids, candidates_am

if __name__=="__main__":
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    livechat = GameBased(
        tokenizer
    )
    print(len(livechat))
    item = livechat[1]
    print(item["video_features"].size())
