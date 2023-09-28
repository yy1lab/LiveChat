import torch
import torch.nn as nn

from models.modules import ContextEncoder, Discriminator, TranscriptEncoder, VideoEncoder

class AVCDiscriminative(nn.Module):
    """
        OUTDATED! Didn't work, to reimplement
    """
    def __init__(
            self, 
            input_size,
            output_size,
            embedding_size,
            hidden_size,
            num_layer_encoder,
            num_layer_decoder,
            enc_dropout,
            dec_dropout,
            batch_first
        ):
        super(AVCDiscriminative, self).__init__()

        self.comment_encoder = ContextEncoder(
            embedding_size, 
            hidden_size, 
            num_layers=num_layer_encoder,
            p=enc_dropout,
            batch_first=batch_first
        )
        self.transcript_encoder = TranscriptEncoder(
            embedding_size, 
            hidden_size, 
            num_layers=num_layer_encoder,
            p=enc_dropout,
            batch_first=batch_first
        )
        self.video_encoder = VideoEncoder(
            2048,
            hidden_size,
            num_layers=num_layer_encoder,
            p=enc_dropout,
            batch_first=batch_first
        )
        self.discriminator = Discriminator(hidden_size)

        self.embeddings = self.comment_encoder.bert.embeddings
        self.transcript_encoder.bert.embeddings = self.comment_encoder.bert.embeddings


    def forward(
            self, 
            input_context_tensor: torch.Tensor, 
            input_transcript_tensor: torch.Tensor, 
            input_video_tensor: torch.Tensor,
            target_tensor: torch.Tensor,
        ):
        hidden_context = self.comment_encoder(input_context_tensor)
        hidden_context = torch.sum(hidden_context, dim=1)

        hidden_transcript = self.transcript_encoder(input_transcript_tensor)

        hidden_video = self.video_encoder(input_video_tensor)
        hidden_video = torch.sum(hidden_video, dim=1)
        
        hidden_target = self.comment_encoder(target_tensor.unsqueeze(1))

        hidden = torch.stack((hidden_context, hidden_transcript.squeeze(1), hidden_video), dim=1)

        output = self.discriminator(hidden, hidden_target)
        output = output.squeeze(1)
        

        return output
