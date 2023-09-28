import os
import heapq
import random
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

from utils import mean_rank, mean_reciprocal_rank, recall

class Trainer():
    def __init__(
            self,
            num_epochs,
            train_loader: DataLoader,
            test_loader: DataLoader,
            eval_loader: DataLoader,
            tokenizer,
            model: nn.Module,
            model_optimizer: optim.Optimizer,
            lr,
            device,
            load='',
            save_dir='model_saves'
    ):
        self.num_epochs = num_epochs
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.eval_loader = eval_loader
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

    def __pretrain_context_encoder_one_batch(self, input_context_tensor, input_context_am, criterion):
        self.model_optimizer.zero_grad()

        inputs, labels = mask_tokens(input_context_tensor, self.tokenizer, self.device)
        decoder_outputs = self.model.context_encoder.pretrain(inputs, input_context_am)
        
        loss=criterion(decoder_outputs.view(-1, self.tokenizer.vocab_size), labels.view(-1))
        loss.backward()
        self.model_optimizer.step()
        return loss.item()
    
    def __pretest_context_encoder(self, criterion):
        test_loss = 0
        for data in tqdm(self.test_loader, desc=' Pretesting context encoder... ', bar_format='{desc} {percentage:3.0f}% - {n_fmt}/{total_fmt}  [{elapsed}, {remaining}]'):       
            chat_context = data["chat_context"].to(self.device)
            chat_context_am = data["chat_context_am"].to(self.device)
            
            with torch.no_grad():
                inputs, labels = mask_tokens(chat_context, self.tokenizer, self.device)
                decoder_outputs = self.model.context_encoder.pretrain(inputs, chat_context_am)
                loss=criterion(decoder_outputs.view(-1, self.tokenizer.vocab_size), labels.view(-1))
                test_loss+=loss.item()
        return test_loss/len(self.test_loader)
    
    def __pretrain_audio_encoder_one_batch(self, transcript_audio, transcript_audio_am, criterion):
        self.model_optimizer.zero_grad()

        inputs, labels = mask_tokens(transcript_audio, self.tokenizer, self.device)
        decoder_outputs = self.model.transcript_encoder.pretrain(inputs, transcript_audio_am)

        loss=criterion(decoder_outputs.view(-1, self.tokenizer.vocab_size), labels.view(-1))
        loss.backward()
        self.model_optimizer.step()
        return loss.item()
    
    def __pretest_audio_encoder(self, criterion):
        test_loss = 0
        for data in tqdm(self.test_loader, desc=' Pretesting audio encoder... ', bar_format='{desc} {percentage:3.0f}% - {n_fmt}/{total_fmt}  [{elapsed}, {remaining}]'):       
            transcript_audio = data["transcript_audio"].to(self.device) if "transcript_audio" in data else torch.Tensor()
            transcript_audio_am = data["transcript_audio_am"].to(self.device) if "transcript_audio_am" in data else torch.Tensor()

            with torch.no_grad():
                inputs, labels = mask_tokens(transcript_audio, self.tokenizer, self.device)
                decoder_outputs = self.model.transcript_encoder.pretrain(inputs, transcript_audio_am)
                loss=criterion(decoder_outputs.view(-1, self.tokenizer.vocab_size), labels.view(-1))
                test_loss+=loss.item()
        return test_loss/len(self.test_loader)

    def __pretrain_decoder_one_batch(self, ):
        ...
    
    def __pretest_decoder(self, ):
        ...
    
    def pretrain(self, criterion):
        """
        Pretrain the model's context encoder and audio encoder.

        During pretraining, the context encoder and audio encoder are trained using masked language modeling.

        Args:
            criterion: The loss criterion for pretraining (e.g., CrossEntropyLoss).

        Note:
            The training progress and loss information will be printed for each epoch.
            The model will be saved after every 20 epochs.

        Returns:
            None
        """
        audio_train_loss = [0 for _ in range(self.epoch)]
        audio_test_loss = [0 for _ in range(self.epoch)]
        
        context_train_loss = [0 for _ in range(self.epoch)]
        context_test_loss = [0 for _ in range(self.epoch)]
        
        decoder_train_loss = [0 for _ in range(self.epoch)]
        decoder_test_loss = [0 for _ in range(self.epoch)]
        
        for epoch in range(self.epoch, self.epoch + self.num_epochs):
            print(f"[Epoch: {epoch+1} / {self.epoch + self.num_epochs}]")
            
            # Pretraining of the audio encoder
            self.model.train()
            train_loss = 0
            for data in tqdm(self.train_loader, desc=' Pretraining audio encoder...', bar_format='{desc} {percentage:3.0f}% - {n_fmt}/{total_fmt}  [{elapsed}, {remaining}]'):
                transcript_audio = data["transcript_audio"].to(self.device) if "transcript_audio" in data else torch.Tensor()
                transcript_audio_am = data["transcript_audio_am"].to(self.device) if "transcript_audio_am" in data else torch.Tensor()

                loss = self.__pretrain_audio_encoder_one_batch(transcript_audio, transcript_audio_am, criterion)
                train_loss += loss
            self.model.eval()
            test_loss = self.__pretest_audio_encoder(criterion)

            print("     Audio pretrain loss: ",train_loss/len(self.train_loader))
            print("     Audio pretest loss: ", test_loss)
            audio_train_loss.append(train_loss/len(self.train_loader))
            audio_test_loss.append(test_loss)
            
            # Pretraining of the context encoder
            self.model.train()
            train_loss = 0
            for data in tqdm(self.train_loader, desc=' Pretraining context encoder...', bar_format='{desc} {percentage:3.0f}% - {n_fmt}/{total_fmt}  [{elapsed}, {remaining}]'):
                chat_context = data["chat_context"].to(self.device)
                chat_context_am = data["chat_context_am"].to(self.device)
                loss = self.__pretrain_context_encoder_one_batch(chat_context, chat_context_am, criterion)
                train_loss += loss
            self.model.eval()
            test_loss = self.__pretest_context_encoder(criterion)

            print("     Context pretrain loss: ",train_loss/len(self.train_loader))
            print("     Context pretest loss: ", test_loss)
            context_train_loss.append(train_loss/len(self.train_loader))
            context_test_loss.append(test_loss)

            # # Pretraining of the decoder
            # self.model.train()
            # train_loss = 0
            # for data in tqdm(self.train_loader, desc=' Pretraining decoder...', bar_format='{desc} {percentage:3.0f}% - {n_fmt}/{total_fmt}  [{elapsed}, {remaining}]'):
            #     responses = data["response"].to(self.device)
            #     responses_am = data["response_am"].to(self.device)
            #     loss = self.__pretrain_decoder_one_batch(responses, responses_am, criterion)
            #     train_loss += loss
            # self.model.eval()
            # test_loss = self.__pretest_decoder(criterion)

            # print("     Decoder pretrain loss: ",train_loss/len(self.train_loader))
            # print("     Decoder pretest loss: ", test_loss)
            # decoder_train_loss.append(train_loss/len(self.train_loader))
            # decoder_test_loss.append(test_loss)


            self.plot_loss(name="pretrain_audio_loss.png", train_loss=audio_train_loss, test_loss=audio_test_loss, epoch=epoch+1)
            self.plot_loss(name="pretrain_context_loss.png", train_loss=context_train_loss, test_loss=context_test_loss, epoch=epoch+1)
            # self.plot_loss(name="pretrain_decoder_loss.png", train_loss=decoder_train_loss, test_loss=decoder_test_loss, epoch=epoch+1)
            if epoch%20==0:
                self.save(self.save_dir, epoch, filename=f"pretrain_{epoch}e.pth")
                print()
        self.save(self.save_dir, -1, filename=f"pretrain_final.pth")


    def __train_one_batch(
            self,
            input_context_tensor,
            input_context_am,
            input_transcript_tensor,
            input_transcript_am,
            input_video_tensor,
            target_tensor,
            target_am,
            criterion
    ):
        self.model_optimizer.zero_grad()

        decoder_outputs = self.model(input_context_tensor, input_context_am, input_transcript_tensor, input_transcript_am, input_video_tensor, target_tensor[:, :-1], target_am[:, :-1])

        one_hot_target = F.one_hot(target_tensor[:, 1:], num_classes = self.tokenizer.vocab_size).to(torch.float)
        loss=criterion(decoder_outputs.permute(0, 2, 1), one_hot_target.permute(0, 2, 1))

        loss.backward()
        self.model_optimizer.step()
        return loss.item()

    def train(self, criterion):
        """
        Train the main generative model.

        Args:
            criterion: The loss criterion for training (e.g., CrossEntropyLoss).

        Note:
            The training progress and loss information will be printed for each epoch.
            The model will be saved after every 5 epochs.

        Returns:
            None
        """
        for epoch in range(self.epoch, self.epoch + self.num_epochs):
            print(f"[Epoch: {epoch+1} / {self.epoch + self.num_epochs}]")
            
            self.model.train()
            train_loss = 0
            for data in tqdm(self.train_loader, desc=' Training...', bar_format='{desc} {percentage:3.0f}% - {n_fmt}/{total_fmt}  [{elapsed}, {remaining}]'):
                chat_context = data["chat_context"].to(self.device)
                chat_context_am = data["chat_context_am"].to(self.device)

                responses = data["response"].to(self.device)
                responses_am = data["response_am"].to(self.device)

                transcript_audio = data["transcript_audio"].to(self.device) if "transcript_audio" in data else torch.Tensor()
                transcript_audio_am = data["transcript_audio_am"].to(self.device) if "transcript_audio_am" in data else torch.Tensor()
                
                video_features = data["video_features"].to(self.device) if "video_features" in data else torch.Tensor()
                category = data["category"].to(self.device) if "category" in data else torch.Tensor()
                
                loss = self.__train_one_batch(chat_context, chat_context_am, transcript_audio, transcript_audio_am, video_features, responses, responses_am, criterion)
                train_loss += loss

            self.model.eval()
            test_loss = self.test(criterion)

            print("     Train loss: ",train_loss/len(self.train_loader))
            print("     Test loss: ", test_loss)
            self.train_loss_track.append(train_loss/len(self.train_loader))
            self.test_loss_track.append(test_loss)

            eval = self.__generate_random(3)
            print()
            for item in eval:
                print("    ", item)
            print()
            self.plot_loss(name="train_test_loss.png", train_loss=self.train_loss_track, test_loss=self.test_loss_track, epoch=epoch+1)
            if epoch%5==0:
                self.save(self.save_dir, epoch)
                print()
        self.save(self.save_dir, epoch)

    def __test_one_batch(
            self,
            input_context_tensor,
            input_context_am,
            input_transcript_tensor,
            input_transcript_am,
            input_video_tensor,
            target_tensor,
            target_am,
            criterion,
    ): 
        with torch.no_grad():
            decoder_outputs = self.model(
                input_context_tensor, 
                input_context_am,
                input_transcript_tensor, 
                input_transcript_am,
                input_video_tensor, 
                target_tensor[:, :-1],
                target_am[:, :-1]
            )
            one_hot_target = F.one_hot(target_tensor[:, 1:], num_classes = self.tokenizer.vocab_size).to(torch.float)
            loss=criterion(decoder_outputs.permute(0, 2, 1), one_hot_target.permute(0, 2, 1))
            
        return loss.item()

    def test(self, criterion):
        """
        Evaluate the model on the test set.

        Args:
            criterion: The loss criterion for evaluation (e.g., CrossEntropyLoss).

        Returns:
            float: Average test loss.
        """
        test_loss = 0
        for data in tqdm(self.test_loader, desc=' Testing... ', bar_format='{desc} {percentage:3.0f}% - {n_fmt}/{total_fmt}  [{elapsed}, {remaining}]'):
            chat_context = data["chat_context"].to(self.device)
            chat_context_am = data["chat_context_am"].to(self.device)
            
            responses = data["response"].to(self.device)
            responses_am = data["response_am"].to(self.device)
            
            transcript_audio = data["transcript_audio"].to(self.device) if "transcript_audio" in data else torch.Tensor()
            transcript_audio_am = data["transcript_audio_am"].to(self.device) if "transcript_audio_am" in data else torch.Tensor()
            
            video_features = data["video_features"].to(self.device) if "video_features" in data else torch.Tensor()
            category = data["category"].to(self.device) if "category" in data else torch.Tensor()
            
            loss = self.__test_one_batch(chat_context, chat_context_am, transcript_audio, transcript_audio_am, video_features, responses, responses_am, criterion)
            test_loss += loss
        return test_loss/len(self.test_loader)
    
    def __eval_one_batch(
            self,
            input_context_tensor,
            input_context_tensor_am,
            input_transcript_tensor,
            input_transcript_tensor_am,
            input_video_tensor,
            candidates_tensor,
            candidates_tensor_am
        ):
        """OUTDATED: used teacher forcing to compute the metrics which was not appropriate
        """
        n_candidates = candidates_tensor.size(1)
        with torch.no_grad():
            hidden_context, hidden_transcript, hidden_video = self.model.encode(
                                                    input_context_tensor, 
                                                    input_context_tensor_am, 
                                                    input_transcript_tensor, 
                                                    input_transcript_tensor_am, 
                                                    input_video_tensor
            )

            candidates_log_likelihoods = []
            for can_id in range(n_candidates):
                current_candidates_tensor = candidates_tensor[:, can_id, :]
                current_candidates_tensor_am = candidates_tensor_am[:, can_id, :]
                decoder_outputs = self.model.decode(hidden_context, hidden_transcript, hidden_video, current_candidates_tensor[:, :-1], current_candidates_tensor_am[:, :-1])
                log_probs = nn.functional.log_softmax(decoder_outputs, dim=2)
                selected_log_probs = torch.gather(log_probs, 2, current_candidates_tensor[:, 1:].unsqueeze(2)).squeeze(2)
                log_likelihoods = torch.sum(selected_log_probs, dim=1)
                candidates_log_likelihoods.append(log_likelihoods)
            
            candidates_log_likelihoods = torch.stack(candidates_log_likelihoods)
            candidates_log_likelihoods = candidates_log_likelihoods.transpose(0, 1)
            indices = torch.argsort(candidates_log_likelihoods, dim=-1, descending=True)
            return indices[:, 0]

    def __eval_one_batch_2(
            self,
            input_context_tensor,
            input_context_tensor_am,
            input_transcript_tensor,
            input_transcript_tensor_am,
            input_video_tensor,
            candidates_tensor,
            candidates_tensor_am
        ):
        n_candidates = candidates_tensor.size(1)
        with torch.no_grad():
            hidden_context, hidden_transcript, hidden_video = self.model.encode(
                                                    input_context_tensor, 
                                                    input_context_tensor_am, 
                                                    input_transcript_tensor, 
                                                    input_transcript_tensor_am, 
                                                    input_video_tensor
            )

            candidates_log_likelihoods = []
            decoded_words, logits = self.__gen_decode(hidden_context, hidden_transcript, hidden_video, candidates_tensor.size(2)-1)

            for can_id in range(n_candidates):
                current_candidates_tensor = candidates_tensor[:, can_id, :]
                # decoder_outputs = self.model.decode(hidden_context, hidden_transcript, hidden_video, current_candidates_tensor[:, :-1])
                log_probs = nn.functional.log_softmax(logits, dim=2)
                selected_log_probs = torch.gather(log_probs, 2, current_candidates_tensor[:, 1:].unsqueeze(2)).squeeze(2)
                log_likelihoods = torch.sum(selected_log_probs, dim=1)
                candidates_log_likelihoods.append(log_likelihoods)
            
            candidates_log_likelihoods = torch.stack(candidates_log_likelihoods)
            candidates_log_likelihoods = candidates_log_likelihoods.transpose(0, 1)
            indices = torch.argsort(candidates_log_likelihoods, dim=-1, descending=True)
            return indices[:, 0]

    def eval(self, ):
        """
        Evaluate the model's performance on the evaluation set.

        Computes various metrics such as Recall@k, Mean Rank, and Mean Reciprocal Rank (MRR).

        Returns:
            dict: A dictionary containing the computed evaluation metrics.
        """
        self.model.eval()

        metrics = {
            "r_at_1": 0,
            "r_at_2": 0,
            "r_at_5": 0,
            "r_at_10": 0,
            "mr": 0,
            "mrr": 0
        }

        for data in tqdm(self.eval_loader, desc=' Evaluating... ', bar_format='{desc} {percentage:3.0f}% - {n_fmt}/{total_fmt}  [{elapsed}, {remaining}]'):
            chat_context = data["chat_context"].to(self.device)
            chat_context_am = data["chat_context_am"].to(self.device)

            transcript_audio = data["transcript_audio"].to(self.device) if "transcript_audio" in data else torch.Tensor()
            transcript_audio_am = data["transcript_audio_am"].to(self.device) if "transcript_audio_am" in data else torch.Tensor()
            
            video_features = data["video_features"].to(self.device) if "video_features" in data else torch.Tensor()
            
            candidates = data["candidates"].to(self.device)
            candidates_am = data["candidates_am"].to(self.device)

            hit_rank = self.__eval_one_batch_2(chat_context, chat_context_am, transcript_audio, transcript_audio_am, video_features, candidates, candidates_am)
            metrics["r_at_1"]+=recall(hit_rank, 1)
            metrics["r_at_2"]+=recall(hit_rank, 2)
            metrics["r_at_5"]+=recall(hit_rank, 5)
            metrics["r_at_10"]+=recall(hit_rank, 10)
            metrics["mr"]+=mean_rank(hit_rank)
            metrics["mrr"]+=mean_reciprocal_rank(hit_rank)

        metrics["r_at_1"]/=len(self.eval_loader)
        metrics["r_at_2"]/=len(self.eval_loader)
        metrics["r_at_5"]/=len(self.eval_loader)
        metrics["r_at_10"]/=len(self.eval_loader)
        metrics["mr"]/=len(self.eval_loader)
        metrics["mrr"]/=len(self.eval_loader)

        print(metrics)
        return metrics
    
    def generate(self, nb_sample):
        """
        Generate comments using the trained model.

        Args:
            nb_sample (int): Number of comments to generate.

        Note:
            The generated comments will be printed along with their corresponding target comments.

        Returns:
            None
        """
        eval = self.__generate_random(nb_sample)
        print()
        for item in eval:
            print("    Target:", item["target"])
            print(" Generated:", item["generated"])
            print()
        print()
    
    def __gen_decode(self, hidden_comments, hidden_audio, hidden_video, max_length, min_length=5):
        with torch.no_grad():  
            batch_size = hidden_comments.size(0)  
            decoder_input = torch.LongTensor([[self.tokenizer.cls_token_id] for _ in range(batch_size)])
            previous = torch.LongTensor([[self.tokenizer.cls_token_id] for _ in range(batch_size)])
            decoder_input = decoder_input.to(self.device)
            decoded_words = []
            logits = []

            for di in range(max_length):
                decoder_output = self.model.generate(
                    hidden_comments,
                    hidden_audio,
                    hidden_video,
                    decoder_input
                )
                if len(decoder_output.size()) == 2:
                    decoder_output = decoder_output.unsqueeze(1)
                _, topi = decoder_output[:, -1, :].data.topk(3)
                current_logits = decoder_output[:, -1, :]
                predicted = topi[:, 0]
                for index, mb_word in enumerate(predicted):
                    if di<min_length:
                        if mb_word.item()==self.tokenizer.sep_token_id:
                            predicted[index]=topi[index, 1]
                    if mb_word.item()==previous[index].item():
                        predicted[index]=topi[index, 2]
                logits.append(current_logits)
                decoded_words.append(predicted)

                decoder_input = torch.cat((decoder_input, predicted.unsqueeze(1).to(self.device)), dim=1)
            
            logits = torch.stack(logits, dim=1)
            decoded_words = torch.stack(decoded_words, dim=1)
            return decoded_words, logits

    def __gen_decode_beam_search(self, hidden_comments, hidden_audio, hidden_video, max_length, beam_width=3):
        with torch.no_grad():
            batch_size = hidden_comments.size(0)
            decoder_input = torch.LongTensor([[self.tokenizer.cls_token_id] for _ in range(batch_size)])
            decoder_input = decoder_input.to(self.device)

            # Initialize beams
            beams = [{'sequence': decoder_input, 'score': 0.0} for _ in range(batch_size)]

            for di in range(max_length):
                all_candidates = []

                for beam in beams:
                    # Get the last token in the sequence for each beam
                    last_token = beam['sequence'][:, -1].unsqueeze(1)

                    # Generate with the current beam sequence
                    decoder_output = self.model.generate(
                        hidden_comments,
                        hidden_audio,
                        hidden_video,
                        beam['sequence']
                    )

                    if len(decoder_output.size()) == 2:
                        decoder_output = decoder_output.unsqueeze(1)

                    # Get the top k tokens and their corresponding log probabilities for each beam
                    topk_log_probs, topk_tokens = decoder_output[:, -1, :].topk(beam_width)

                    for k in range(beam_width):
                        # Create a new beam for each top-k token
                        new_sequence = torch.cat((beam['sequence'], topk_tokens[:, k].unsqueeze(1).to(self.device)), dim=1)
                        new_score = beam['score'] + topk_log_probs[:, k]

                        # Calculate logits for the new sequence
                        logits = self.model.generate(hidden_comments, hidden_audio, hidden_video, new_sequence)[-1]

                        all_candidates.append({'sequence': new_sequence, 'score': new_score, 'logits': logits})

                # Select the top-k candidates based on their scores
                ordered_candidates = sorted(all_candidates, key=lambda x: x['score'].sum(), reverse=True)
                beam = ordered_candidates[1]

            # Get the final decoded sequences and their corresponding logits
            decoded_sentence = beam['sequence']
            logits = beam['logits']

            return decoded_sentence, logits

    def __generate_random(self, nb_eval):
        max_length = 20
        max_index = len(self.test_loader.dataset)
        indexes=[]
        for _ in range(nb_eval):
            indexes.append(random.randint(0, max_index-1))
        gen = []
        for i in range(nb_eval):
            dataset_element = self.eval_loader.dataset[indexes[i]]
            chat_context = dataset_element["chat_context"].to(self.device)
            chat_context_am = dataset_element["chat_context_am"].to(self.device)

            transcript_audio = dataset_element["transcript_audio"].to(self.device) if "transcript_audio" in dataset_element else torch.Tensor().to(self.device)
            transcript_audio_am = dataset_element["transcript_audio_am"].to(self.device) if "transcript_audio_am" in dataset_element else torch.Tensor().to(self.device)
            
            video_features = dataset_element["video_features"].to(self.device) if "video_features" in dataset_element else torch.Tensor().to(self.device)
            target = self.tokenizer.decode(dataset_element["response"][1:], skip_special_tokens=True)
            with torch.no_grad():
                hidden_comments, hidden_audio, hidden_video = self.model.encode(
                                                    chat_context.unsqueeze(0), 
                                                    chat_context_am.unsqueeze(0), 
                                                    transcript_audio.unsqueeze(0), 
                                                    transcript_audio_am.unsqueeze(0), 
                                                    video_features.unsqueeze(0)
                )
                
                response, logits = self.__gen_decode(hidden_comments, hidden_audio, hidden_video, max_length)
                response = self.tokenizer.decode(response.squeeze(), skip_special_tokens=True)

            gen.append({"target": target, "generated": response})
        return gen

    def plot_loss(self, name, train_loss, test_loss=None, epoch=None):
        """
        Plot the training and test losses.

        Args:
            name (str): The name of the plot file to be saved.
            train_loss (list): List of training losses over epochs.
            test_loss (list, optional): List of test losses over epochs. Defaults to None.
            epoch (int, optional): The current epoch number. Defaults to None.

        Returns:
            None
        """
        if epoch==None:
            epoch=self.epoch
        plt.plot(range(epoch), train_loss, color="r", label="train")
        if test_loss!=None:
            plt.plot(range(epoch), test_loss, color="b", label="test")
        plt.title(f"Train and Test Losses on {epoch-1} epochs")
        plt.legend()
        plt.savefig(name)
        plt.clf()

    def load(self, dir, filename):
        """
        Load a saved model checkpoint.

        Args:
            dir (str): Directory where the model checkpoint is saved.
            filename (str): Name of the model checkpoint file.

        Returns:
            None
        """
        print(f'Loading {filename}...')
        checkpoint = torch.load(os.path.join(dir, filename))
        self.epoch = checkpoint['epoch']+1
        self.model.load_state_dict(checkpoint['model'])
        self.model_optimizer.load_state_dict(checkpoint['model_optimizer'])

        self.train_loss_track = checkpoint["train_loss_track"]
        self.test_loss_track = checkpoint["test_loss_track"]

    def save(self, dir, epoch, filename=None):
        """
        Save the current model checkpoint.

        Args:
            dir (str): Directory to save the model checkpoint.
            epoch (int): The current epoch number.
            filename (str, optional): Name of the model checkpoint file. If None, a default filename will be used.

        Returns:
            None
        """
        if filename==None:
            filename = f'checkpoint_{self.lr}_{epoch}e.pth'
        print('Saving model...')

        torch.save(
            {
            'epoch': epoch,

            'model': self.model.state_dict(),
            'model_optimizer': self.model_optimizer.state_dict(),

            'train_loss_track': self.train_loss_track,
            'test_loss_track': self.test_loss_track
            },
            os.path.join(dir, filename)
        )

def start(
        model: nn.Module,
        num_epochs: int,
        learning_rate: float,
        batch_size: int,
        filename_model: str,
        device: torch.device,
        mode: str,
        num_workers: int,
        tokenizer: AutoTokenizer,
        train_dataset: torch.utils.data.Dataset,
        test_dataset: torch.utils.data.Dataset,
        eval_dataset: torch.utils.data.Dataset,
        save_dir: str
):
    """
    Start the training, evaluation, or generation process for the generative model.

    Args:
        model (nn.Module): The generative model to be trained, evaluated, or used for generation.
        num_epochs (int): Number of epochs to train the model.
        learning_rate (float): Learning rate used by the optimizer during training.
        batch_size (int): Batch size used for data loading during training, evaluation, or generation.
        filename_model (str): Name of the model checkpoint file to load, if available.
        device (torch.device): Torch device (e.g., 'cuda', 'cpu') to place the model and data on.
        mode (str): The mode of operation, either 'train', 'eval', 'gen', or 'pretrain'.
        num_workers (int): Number of worker processes used for data loading.
        tokenizer (AutoTokenizer): Tokenizer to encode and decode text sequences.
        train_dataset (torch.utils.data.Dataset): Dataset containing training data.
        test_dataset (torch.utils.data.Dataset): Dataset containing test data.
        eval_dataset (torch.utils.data.Dataset): Dataset containing evaluation data.
        save_dir (str): Directory to save model checkpoints.

    Note:
        The function will initialize the Trainer object and perform the selected operation based on the 'mode' parameter.

    Returns:
        None
    """
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    model_optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    pretrain_criterion = nn.CrossEntropyLoss(ignore_index=-100)

    trainer = Trainer(
        num_epochs,
        train_loader,
        test_loader,
        eval_loader,
        tokenizer,
        model,
        model_optimizer,
        learning_rate,
        device,
        load=filename_model,
        save_dir=save_dir
    )
    if mode=="train":
        trainer.train(criterion)
    elif mode=="eval":
        trainer.eval()
    elif mode=="gen":
        trainer.generate(10)
    elif mode=="pretrain":
        trainer.pretrain(pretrain_criterion)

def mask_tokens(inputs, tokenizer, device, masking_prob=0.2):
    """
    Apply masking to a batch of input tokens for masked language modeling.

    Args:
        inputs (torch.Tensor): Input tokens to be masked.
        tokenizer (AutoTokenizer): Tokenizer used to encode the input tokens.
        device: Torch device (e.g., 'cuda', 'cpu') to place the data on.
        masking_prob (float, optional): Probability of masking each token. Defaults to 0.2.

    Returns:
        torch.Tensor: The masked input tokens.
        torch.Tensor: The masked labels used for computing the masked language modeling loss.
    """
    labels = inputs.clone().to(device)
    mask_indices = torch.bernoulli(torch.full(labels.shape, masking_prob)).bool().to(device)
    labels[~mask_indices] = -100
    masked_indices = mask_indices & (inputs != tokenizer.pad_token_id)
    inputs[masked_indices] = tokenizer.mask_token_id
    return inputs, labels
