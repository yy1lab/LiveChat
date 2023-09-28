import os
import torch
from transformers import logging, AutoTokenizer
from game_based_dataset import GameBased
from livechat import StreamChatDataset
from models.avc_generative import AVCGenerative
from models.vc_generative import VCGenerative

import train.trainer as trainer
from utils import parse_args

if __name__=="__main__":
    """
    Main script for training generative dialogue models.

    This script initializes the necessary components for training a generative dialogue model and starts the training process.

    It sets up hyperparameters, loads the specified dataset, initializes the model, and starts the training using the Trainer class.

    Example:
        To train an AVCGenerative model on the 'livechat' dataset:
        python script_name.py --model avc --d livechat --e 100 -lr 1e-5 -b 32 -l model_save_name.pth -m train
    """
    args=parse_args()
    
    # Setting up hyperparameters
    logging.set_verbosity_error()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    num_epochs = args.e
    learning_rate = args.lr
    batch_size = args.b
    filename_model = args.l
    mode = args.m
    dataset = args.d
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = 8
    input_size = 30522
    embedding_size = 256
    hidden_size = 256
    output_size = 30522
    num_layers_encoder = 4
    num_layers_decoder = 4
    enc_dropout = 0.1
    dec_dropout = 0.1
    weight_decay = 0.01
    comments_padding = 10
    transcript_padding = 100
    candidates_padding = 5
    nb_context_comments = 5
    nb_context_comments_eval = 15
    tokenizer = AutoTokenizer.from_pretrained('prajjwal1/bert-mini')

    
    # Loading the dataset
    if dataset=="livechat":
        train_file = "train_reduced.json"
        test_file = "test_reduced.json"
        eval_file = "test_reduced_candidates.json"

        train_dataset = StreamChatDataset(
            tokenizer, 
            "/media/livechat/dataset/", 
            "features/", 
            train_file, 
            comments_padding=comments_padding,
            transcript_padding=transcript_padding,
            nb_context_comments=nb_context_comments
        )
        test_dataset = StreamChatDataset(
            tokenizer, 
            "/media/livechat/dataset/", 
            "features/", 
            test_file, 
            mode="train",
            comments_padding=comments_padding,
            transcript_padding=transcript_padding,
            nb_context_comments=nb_context_comments
        )
        eval_dataset = StreamChatDataset(
            tokenizer, 
            "/media/livechat/dataset/", 
            "features/", 
            eval_file, 
            mode="eval",
            comments_padding=comments_padding,
            transcript_padding=transcript_padding,
            candidates_padding=candidates_padding,
            nb_context_comments=nb_context_comments_eval
        )
    elif dataset=="gdialogue":
        train_file = "train.json"
        test_file = "test.json"
        eval_file = "val.json"
        
        train_dataset = GameBased(
            tokenizer, 
            "/media/livechat/game_based_dialogue/", 
            train_file,
            "train_video_feat.h5",
            comments_padding=comments_padding,
            nb_context_comments=nb_context_comments,
            mode="train"
        )
        test_dataset = GameBased(
            tokenizer,
            "/media/livechat/game_based_dialogue/", 
            test_file,
            "test_video_feat.h5",
            comments_padding=comments_padding,
            nb_context_comments=nb_context_comments,
            mode="eval"
        )
        eval_dataset = GameBased(
            tokenizer, 
            "/media/livechat/game_based_dialogue/", 
            eval_file,
            "val_video_feat.h5",
            comments_padding=comments_padding,
            nb_context_comments=nb_context_comments,
            mode="eval"
        )
    
    # Loading the model
    if args.model=="avc":
        model = AVCGenerative(
            input_size=input_size,
            output_size=output_size,
            embedding_size=embedding_size,
            hidden_size=hidden_size,
            num_layer_encoder=num_layers_encoder,
            num_layer_decoder=num_layers_decoder,
            enc_dropout=enc_dropout,
            dec_dropout=dec_dropout,
            batch_first=True
        )
        save_dir="/media/livechat/model_saves/avc_transformer"
    elif args.model == "vc":
        model = VCGenerative(
            input_size=input_size,
            output_size=output_size,
            embedding_size=embedding_size,
            hidden_size=hidden_size,
            num_layer_encoder=num_layers_encoder,
            num_layer_decoder=num_layers_decoder,
            enc_dropout=enc_dropout,
            dec_dropout=dec_dropout,
            batch_first=True
        )
        save_dir="/media/livechat/model_saves/game_based"


    trainer.start(
        model,
        num_epochs,
        learning_rate,
        batch_size,
        filename_model,
        device,
        mode,
        num_workers,
        tokenizer,
        train_dataset,
        test_dataset,
        eval_dataset,
        save_dir
    )