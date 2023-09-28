import torch
import torch.nn as nn

from models.modules import CommentDecoder, ContextEncoder, TranscriptEncoder, VideoEncoder

class AVCGenerative(nn.Module):
    """
    An end-to-end generative model that combines context, transcript, and video information to generate comments.

    Args:
        input_size (int): The size of the input features for context, transcript, and video.
        output_size (int): The size of the output comment features.
        embedding_size (int): The size of the embeddings for input sequences.
        hidden_size (int): The number of features in the hidden layer for the encoder and decoder.
        num_layer_encoder (int): The number of layers in the encoder.
        num_layer_decoder (int): The number of layers in the decoder.
        enc_dropout (float): The dropout probability for the encoder.
        dec_dropout (float): The dropout probability for the decoder.
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
        super(AVCGenerative, self).__init__()


        self.context_encoder = ContextEncoder(
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
        
        self.comment_decoder = CommentDecoder(
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

        self.transcript_encoder.bert.embeddings = self.context_encoder.bert.embeddings

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
        Forward pass of the AVCGenerative model.

        Args:
            input_context_tensor (torch.Tensor): Tensor representing the input context features of shape (batch_size, num_modalities, context_features).
            input_context_am (torch.Tensor): Attention mask for the context input tensor, of shape (batch_size, num_modalities, context_length).
            input_transcript_tensor (torch.Tensor): Tensor representing the input transcript features of shape (batch_size, transcript_length, transcript_features).
            input_transcript_am (torch.Tensor): Attention mask for the transcript input tensor, of shape (batch_size, transcript_length).
            input_video_tensor (torch.Tensor): Tensor representing the input video features of shape (batch_size, video_features).
            target_tensor (torch.Tensor): Target tensor for the decoder input of shape (batch_size, target_length).
            target_am (torch.Tensor): Attention mask for the target input tensor, of shape (batch_size, target_length).

        Returns:
            torch.Tensor: Output tensor containing the predicted comment features, of shape (batch_size, target_length, output_size).
        """
        if self.target_mask==None:
            self.__init_target_mask(target_tensor.size(1), target_tensor.device)

        hidden_context = self.context_encoder(input_context_tensor, input_context_am)

        hidden_transcript = self.transcript_encoder(input_transcript_tensor, input_transcript_am)

        hidden_video = self.video_encoder(input_video_tensor)

        emb_target_comment = self.embeddings(target_tensor)
        decoder_outputs = self.comment_decoder(emb_target_comment, hidden_context, hidden_transcript, hidden_video, self.target_mask, target_am)

        return decoder_outputs
    
    def encode(
            self, 
            input_context_tensor: torch.Tensor, 
            input_context_am: torch.Tensor, 
            input_transcript_tensor: torch.Tensor, 
            input_transcript_am: torch.Tensor, 
            input_video_tensor: torch.Tensor
        ):
        """
        Encode input tensors using the context, transcript, and video encoders.

        Args:
            input_context_tensor (torch.Tensor): Tensor representing the input context features of shape (batch_size, num_modalities, context_features).
            input_context_am (torch.Tensor): Attention mask for the context input tensor, of shape (batch_size, num_modalities, context_length).
            input_transcript_tensor (torch.Tensor): Tensor representing the input transcript features of shape (batch_size, transcript_length, transcript_features).
            input_transcript_am (torch.Tensor): Attention mask for the transcript input tensor, of shape (batch_size, transcript_length).
            input_video_tensor (torch.Tensor): Tensor representing the input video features of shape (batch_size, video_features).

        Returns:
            torch.Tensor: Hidden context features of shape (batch_size, num_modalities, hidden_size).
            torch.Tensor: Hidden transcript features of shape (batch_size, transcript_length, hidden_size).
            torch.Tensor: Hidden video features of shape (batch_size, video_features, hidden_size).
        """
        hidden_context = self.context_encoder(input_context_tensor, input_context_am)

        hidden_transcript = self.transcript_encoder(input_transcript_tensor, input_transcript_am)

        hidden_video = self.video_encoder(input_video_tensor)

        return hidden_context, hidden_transcript, hidden_video
    
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
            hidden_audio (torch.Tensor): Hidden transcript features of shape (batch_size, transcript_length, hidden_size).
            hidden_video (torch.Tensor): Hidden video features of shape (batch_size, video_features, hidden_size).
            target_tensor (torch.Tensor): Target tensor for the decoder input of shape (batch_size, target_length).
            target_am (torch.Tensor): Attention mask for the target input tensor, of shape (batch_size, target_length).

        Returns:
            torch.Tensor: Output tensor containing the predicted comment features, of shape (batch_size, target_length, output_size).
        """
        emb_target_comment = self.embeddings(target_tensor)
        decoder_outputs = self.comment_decoder(emb_target_comment, hidden_comments, hidden_audio, hidden_video, self.target_mask, target_am)

        return decoder_outputs
    
    def generate(
            self,
            hidden_comments: torch.Tensor,
            hidden_audio: torch.Tensor,
            hidden_video: torch.Tensor,
            target_tensor: torch.Tensor
        ):
        """
        Generate comments using the comment decoder.

        Args:
            hidden_comments (torch.Tensor): Hidden context features of shape (batch_size, num_modalities, hidden_size).
            hidden_audio (torch.Tensor): Hidden transcript features of shape (batch_size, transcript_length, hidden_size).
            hidden_video (torch.Tensor): Hidden video features of shape (batch_size, video_features, hidden_size).
            target_tensor (torch.Tensor): Target tensor for the decoder input of shape (batch_size, target_length).

        Returns:
            torch.Tensor: Output tensor containing the predicted comment features, of shape (batch_size, target_length, output_size).
        """
        emb_target_comment = self.embeddings(target_tensor)
        decoder_outputs = self.comment_decoder(emb_target_comment, hidden_comments, hidden_audio, hidden_video)

        return decoder_outputs
    
    def __init_target_mask(self, tg_length, device):
        """
        Initialize the target mask for the decoder.

        Args:
            tg_length (int): The length of the target tensor.
            device (torch.device): The device on which the target mask should be initialized.
        """
        upper_triangular_mask = torch.triu(torch.full((tg_length, tg_length), float('-inf')), diagonal=1)
        self.target_mask = upper_triangular_mask.to(device)
        self.target_mask = self.target_mask.bool()
