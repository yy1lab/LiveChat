import os
import heapq
import random
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
from livechat import StreamChatDataset
from models.models import AVCDiscriminative, AVCGenerative
from utils import mean_rank, mean_reciprocal_rank, recall

class Trainer():
    def __init__(
            self,
            num_epochs,
            train_loader: DataLoader,
            test_loader: DataLoader,
            tokenizer,
            model: AVCGenerative,
            model_optimizer: optim.Optimizer,
            lr,
            device,
            load='',
            save_dir='model_saves'
    ):
        self.num_epochs = num_epochs
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.tokenizer = tokenizer
        self.model = model.to(device)
        self.model_optimizer = model_optimizer
        self.lr = lr
        self.device = device

        self.save_dir = save_dir
        self.epoch = 0
        self.train_loss_track = []
        self.test_loss_track = []
        if load!='':
            self.load(save_dir, load)

    def __train_one_batch(
            self,
            input_context_tensor,
            input_transcript_tensor,
            input_video_tensor,
            target_tensor,
            labels_tensor,
            criterion
    ):
        self.model_optimizer.zero_grad()

        logits = self.model(input_context_tensor, input_transcript_tensor, input_video_tensor, target_tensor)
        loss=criterion(logits, labels_tensor.float())

        loss.backward()
        self.model_optimizer.step()

        return loss.item()

    def train(self, criterion):
        for epoch in range(self.epoch, self.epoch + self.num_epochs):
            print(f"[Epoch: {epoch+1} / {self.epoch + self.num_epochs}]")
            
            self.model.train()
            train_loss = 0
            for data in tqdm(self.train_loader, desc=' Training...', bar_format='{desc} {percentage:3.0f}% - {n_fmt}/{total_fmt}  [{elapsed}, {remaining}]'):
                chat_context = data["chat_context"].to(self.device)
                response = data["response_placeholder"].to(self.device)
                transcript_audio = data["transcript_audio"].to(self.device)
                video_features = data["video_features"].to(self.device)
                category = data["category"].to(self.device)
                labels = data["label"].to(self.device)
                loss = self.__train_one_batch(chat_context, transcript_audio, video_features, response, labels, criterion)
                train_loss += loss

            self.model.eval()
            test_loss = self.test(criterion)

            print("     Train loss: ",train_loss/len(self.train_loader))
            print("     Test loss: ", test_loss)
            self.train_loss_track.append(train_loss/len(self.train_loader))
            self.test_loss_track.append(test_loss)

            self.plot_loss(epoch=epoch+1)
            if epoch%5==0:
                self.save(self.save_dir, epoch, train_loss/len(self.train_loader))
                print()
        self.save(self.save_dir, epoch, train_loss/len(self.train_loader))

    def __test_one_batch(
            self,
            input_context_tensor,
            input_transcript_tensor,
            input_video_tensor,
            target_tensor,
            labels_tensor,
            criterion,
    ): 
        with torch.no_grad():
            logits = self.model(input_context_tensor, input_transcript_tensor, input_video_tensor, target_tensor)
            loss=criterion(logits, labels_tensor.float())
        return loss.item()

    def test(self, criterion):
        test_loss = 0
        for data in tqdm(self.test_loader, desc=' Testing... ', bar_format='{desc} {percentage:3.0f}% - {n_fmt}/{total_fmt}  [{elapsed}, {remaining}]'):
            chat_context = data["chat_context"].to(self.device)
            responses = data["response_placeholder"].to(self.device)
            transcript_audio = data["transcript_audio"].to(self.device)
            video_features = data["video_features"].to(self.device)
            category = data["category"].to(self.device)
            labels = data["label"].to(self.device)
            loss = self.__test_one_batch(chat_context, transcript_audio, video_features, responses, labels, criterion)
            test_loss += loss
        return test_loss/len(self.test_loader)
    
    def __eval_one_batch(
            self,
            input_context_tensor,
            input_transcript_tensor,
            input_video_tensor,
            candidates_tensor
        ):
        n_candidates = candidates_tensor.size(1)
        with torch.no_grad():
            hidden = self.model.encode(input_context_tensor, input_transcript_tensor, input_video_tensor)

            candidates_log_likelihoods = []
            for can_id in range(n_candidates):
                current_candidates_tensor = candidates_tensor[:, can_id, :]
                decoder_outputs = self.model.decode(hidden, current_candidates_tensor[:, :-1])
                log_probs = nn.functional.log_softmax(decoder_outputs, dim=2)
                selected_log_probs = torch.gather(log_probs, 2, current_candidates_tensor[:, 1:].unsqueeze(2)).squeeze(2)
                log_likelihoods = torch.sum(selected_log_probs, dim=1)
                candidates_log_likelihoods.append(log_likelihoods)
            
            candidates_log_likelihoods = torch.stack(candidates_log_likelihoods)
            candidates_log_likelihoods = candidates_log_likelihoods.transpose(0, 1)
            indices = torch.argsort(candidates_log_likelihoods, dim=-1, descending=True)
            return indices[:, 0]

    def eval(self, ):
        self.model.eval()

        metrics = {
            "r_at_1": 0,
            "r_at_5": 0,
            "r_at_10": 0,
            "mr": 0,
            "mrr": 0
        }

        for data in tqdm(self.test_loader, desc=' Evaluating... ', bar_format='{desc} {percentage:3.0f}% - {n_fmt}/{total_fmt}  [{elapsed}, {remaining}]'):
            chat_context = data["chat_context"].to(self.device)
            responses = data["response"].to(self.device)
            transcript_audio = data["transcript_audio"].to(self.device)
            video_features = data["video_features"].to(self.device)
            category = data["category"].to(self.device)
            candidates = data["candidates"].to(self.device)

            hit_rank = self.__eval_one_batch(chat_context, transcript_audio, video_features, candidates)
            metrics["r_at_1"]+=recall(hit_rank, 1)
            metrics["r_at_5"]+=recall(hit_rank, 5)
            metrics["r_at_10"]+=recall(hit_rank, 10)
            metrics["mr"]+=mean_rank(hit_rank)
            metrics["mrr"]+=mean_reciprocal_rank(hit_rank)

        metrics["r_at_1"]/=len(self.test_loader)
        metrics["r_at_5"]/=len(self.test_loader)
        metrics["r_at_10"]/=len(self.test_loader)
        metrics["mr"]/=len(self.test_loader)
        metrics["mrr"]/=len(self.test_loader)

        print(metrics)

        return metrics
        
    def plot_loss(self, epoch=None):
        if epoch==None:
            epoch=self.epoch
        plt.plot(range(epoch), self.train_loss_track, color="r", label="train")
        plt.plot(range(epoch), self.test_loss_track, color="b", label="test")
        plt.title(f"Train and Test Losses on {epoch-1} epochs")
        plt.legend()
        plt.savefig("train_test_loss_disc.png")
        plt.clf()

    def load(self, dir, filename):
        print(f'Loading {filename}...')
        checkpoint = torch.load(os.path.join(dir, 'discriminative', filename))

        self.epoch = checkpoint['epoch']+1
        self.model.load_state_dict(checkpoint['model'])
        self.model_optimizer.load_state_dict(checkpoint['model_optimizer'])

        self.train_loss_track = checkpoint["train_loss_track"]
        self.test_loss_track = checkpoint["test_loss_track"]

    def save(self, dir, epoch, loss):
        print('Saving model...')

        torch.save(
            {
            'epoch': epoch,

            'model': self.model.state_dict(),
            'model_optimizer': self.model_optimizer.state_dict(),

            'train_loss_track': self.train_loss_track,
            'test_loss_track': self.test_loss_track
            },
            os.path.join(dir, 'discriminative', f'checkpoint_{self.lr}_{self.train_loader.dataset.opti}_{epoch}e.pth')
        )

def train(args):
    num_epochs = args.e
    learning_rate = args.lr
    batch_size = args.b
    filename_model = args.l
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode = args.m
    
    num_workers = 8
    input_size = 30522
    embedding_size = 256
    hidden_size = 256
    output_size = 30522
    num_layers_encoder = 1
    num_layers_decoder = 1
    enc_dropout = 0.2
    dec_dropout = 0.2
    batch_first = True
    comments_padding = 50
    transcript_padding = 100
    nb_context_comments = 10
    opti_mode = "gen"
    train_file = "train_reduced.json"
    test_file = "test_reduced.json"
    eval_file = "test_fortnite_candidates.json"

    tokenizer = AutoTokenizer.from_pretrained('prajjwal1/bert-mini')
    
    train_dataset = StreamChatDataset(
        tokenizer, 
        "/media/livechat/dataset/", 
        "features/", 
        train_file, 
        comments_padding=comments_padding,
        transcript_padding=transcript_padding,
        nb_context_comments=nb_context_comments,
        opti=opti_mode
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    test_dataset = StreamChatDataset(
        tokenizer, 
        "/media/livechat/dataset/", 
        "features/", 
        test_file if mode=="train" else eval_file, 
        mode=mode,
        comments_padding=comments_padding,
        transcript_padding=transcript_padding,
        nb_context_comments=nb_context_comments,
        opti=opti_mode
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    model = AVCDiscriminative(
        input_size=input_size,
        output_size=output_size,
        embedding_size=embedding_size,
        hidden_size=hidden_size,
        num_layer_encoder=num_layers_encoder,
        num_layer_decoder=num_layers_decoder,
        enc_dropout=enc_dropout,
        dec_dropout=dec_dropout,
        batch_first=batch_first
    )
    
    model_optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    criterion = nn.BCELoss()

    trainer = Trainer(
        num_epochs,
        train_loader,
        test_loader,
        tokenizer,
        model,
        model_optimizer,
        learning_rate,
        device,
        load=filename_model,
        save_dir="/media/livechat/model_saves"
    )
    if mode=="train":
        trainer.train(criterion)
    elif mode=="eval":
        trainer.eval()
