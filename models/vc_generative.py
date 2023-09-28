import torch
import torch.nn as nn

from models.modules import ContextEncoder, VCDecoder, VideoEncoder

class VCGenerative(nn.Module):
    """
    Video-Context Generative Model (VCGenerative).

    This model takes input tensors representing video features, context, and transcript, and generates
    comments based on the inputs. It consists of a context encoder, a video encoder, and a comment decoder.

    Args:
        input_size (int): Size of the input tensors.
        output_size (int): Size of the output tensors.
        embedding_size (int): Size of the embedding layer.
        hidden_size (int): Size of the hidden layers.
        num_layer_encoder (int): Number of layers in the encoder.
        num_layer_decoder (int): Number of layers in the decoder.
        enc_dropout (float): Dropout rate for the encoder.
        dec_dropout (float): Dropout rate for the decoder.
        batch_first (bool): If True, the batch dimension is the first dimension; otherwise, it is the second dimension.
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
        super(VCGenerative, self).__init__()


        self.context_encoder = ContextEncoder(
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
        
        self.comment_decoder = VCDecoder(
            embedding_size, 
            hidden_size, 
            output_size, 
            n_head=8,
            dim_ff=512,
            num_layers=num_layer_decoder, 
            p=dec_dropout,
            batch_first=batch_first
        )

        self.embeddings = self.context_encoder.bert.embeddings

        self.target_mask = None

    def forward(
            self, 
            input_context_tensor: torch.Tensor, 
            input_context_am: torch.Tensor, 
            input_transcript_tensor: torch.Tensor, 
            input_transcript_am: torch.Tensor, 
            input_video_tensor: torch.Tensor,
            target_tensor: torch.Tensor,
            target_am: torch.Tensor
        ):
        """
        Forward pass of the VCGenerative model.

        Args:
            input_context_tensor (torch.Tensor): Input context tensor of shape (batch_size, context_length, input_size).
            input_context_am (torch.Tensor): Input attention mask tensor for context of shape (batch_size, context_length, input_size).
            input_transcript_tensor (torch.Tensor): Input transcript tensor of shape (batch_size, transcript_length, input_size).
            input_transcript_am (torch.Tensor): Input attention mask tensor for transcript of shape (batch_size, transcript_length, input_size).
            input_video_tensor (torch.Tensor): Input video tensor of shape (batch_size, video_features, input_size).
            target_tensor (torch.Tensor): Target tensor for the decoder input of shape (batch_size, target_length).
            target_am (torch.Tensor): Target attention mask tensor of shape (batch_size, target_length, target_length).

        Returns:
            torch.Tensor: Output tensor containing the predicted comment features, of shape (batch_size, target_length, output_size).
        """
        if self.target_mask==None:
            self.__init_target_mask(target_tensor.size(1), target_tensor.device)

        hidden_context = self.context_encoder(input_context_tensor)

        hidden_video = self.video_encoder(input_video_tensor)

        emb_target_comment = self.embeddings(target_tensor)
        decoder_outputs = self.comment_decoder(emb_target_comment, hidden_context, hidden_video, self.target_mask)

        return decoder_outputs
    
    def encode(
            self, 
            input_context_tensor: torch.Tensor, 
            input_context_am: torch.Tensor, 
            input_audio_tensor: torch.Tensor,
            input_audio_am: torch.Tensor,
            input_video_tensor: torch.Tensor
        ):
        """
        Encode input tensors using the context and video encoders.

        Args:
            input_context_tensor (torch.Tensor): Input context tensor of shape (batch_size, context_length, input_size).
            input_context_am (torch.Tensor): Input attention mask tensor for context of shape (batch_size, context_length, input_size).
            input_audio_tensor (torch.Tensor): Input audio tensor of shape (batch_size, audio_length, input_size).
            input_audio_am (torch.Tensor): Input attention mask tensor for audio of shape (batch_size, audio_length, input_size).
            input_video_tensor (torch.Tensor): Input video tensor of shape (batch_size, video_features, input_size).

        Returns:
            tuple: A tuple containing the hidden context, hidden audio (an empty tensor), and hidden video, respectively.
        """
        hidden_context = self.context_encoder(input_context_tensor)


        hidden_video = self.video_encoder(input_video_tensor)

        return hidden_context, torch.Tensor(), hidden_video
    
    def decode(
            self,
            hidden_comments: torch.Tensor,
            hidden_audio: torch.Tensor,
            hidden_video: torch.Tensor,
            target_tensor: torch.Tensor,
            target_am: torch.Tensor
        ):
        """
        Decode input tensors using the comment decoder.

        Args:
            hidden_comments (torch.Tensor): Hidden context features of shape (batch_size, num_modalities, hidden_size).
            hidden_audio (torch.Tensor): Empty tensor (not used).
            hidden_video (torch.Tensor): Hidden video features of shape (batch_size, video_features, hidden_size).
            target_tensor (torch.Tensor): Target tensor for the decoder input of shape (batch_size, target_length).
            target_am (torch.Tensor): Target attention mask tensor of shape (batch_size, target_length, target_length).

        Returns:
            torch.Tensor: Output tensor containing the predicted comment features, of shape (batch_size, target_length, output_size).
        """
        if self.target_mask==None:
            self.__init_target_mask(target_tensor.size(1), target_tensor.device)

        emb_target_comment = self.embeddings(target_tensor)
        decoder_outputs = self.comment_decoder(emb_target_comment, hidden_comments, hidden_video, self.target_mask)

        return decoder_outputs
    
    def generate(
            self,
            hidden_comments: torch.Tensor,
            hidden_audio: torch.Tensor,
            hidden_video: torch.Tensor,
            target_tensor: torch.Tensor
        ):
        """
        Generate comments based on the input tensors.

        Args:
            hidden_comments (torch.Tensor): Hidden state for the comments.
            hidden_audio (torch.Tensor): Hidden state for the audio.
            hidden_video (torch.Tensor): Hidden state for the video.
            target_tensor (torch.Tensor): Input tensor representing the target.

        Returns:
            torch.Tensor: Generated comments.
        """
        emb_target_comment = self.embeddings(target_tensor)
        decoder_outputs = self.comment_decoder(emb_target_comment, hidden_comments, hidden_video)

        return decoder_outputs

    def __init_target_mask(self, tg_length, device):
        """
        Initialize the target mask for self-attention.

        Args:
            tg_length (int): Target length.
            device (torch.device): Device on which the mask will be initialized.
        """
        upper_triangular_mask = torch.triu(torch.full((tg_length, tg_length), float('-inf')), diagonal=1)
        self.target_mask = upper_triangular_mask.to(device)
