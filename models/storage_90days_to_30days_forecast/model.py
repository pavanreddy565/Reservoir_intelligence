import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size=50):
        super().__init__()
        self.gru = nn.LSTM(input_size=input_size, hidden_size=hidden_size, 
                          batch_first=True, bidirectional=False)
    
    def forward(self, x):
        # Output contains all hidden states; hidden is the last hidden state
        outputs, hidden = self.gru(x)
        return outputs, hidden

class Decoder(nn.Module):
    def __init__(self, hidden_size=50, output_size=1):
        super().__init__()
        self.gru = nn.LSTM(input_size=1, hidden_size=hidden_size, 
                          batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.drop = nn.Dropout(p=0.2)
    
    def forward(self, x, hidden):
        # Decode one time step
        x, hidden = self.gru(x, hidden)
        x = self.drop(x)
        output = self.fc(x)
        return output, hidden

class EncoderDecoder(nn.Module):
    def __init__(self, input_size, hidden_size=50, lookahead=1):
        super().__init__()
        self.encoder = Encoder(input_size, hidden_size)
        self.decoder = Decoder(hidden_size, output_size=1)
        self.lookahead = lookahead
    
    def forward(self, x, target_seq=None, teacher_forcing_ratio=0.5):
        # Encode the input sequence
        _, hidden = self.encoder(x)
        
        # Initialize the decoder input with zeros
        decoder_input = torch.zeros((x.size(0), 1, 1), device=x.device)
        
        outputs = []
        for t in range(self.lookahead):
            # Decode each time step
            decoder_output, hidden = self.decoder(decoder_input, hidden)
            outputs.append(decoder_output)
            
            # Teacher forcing
            if target_seq is not None and torch.rand(1).item() < teacher_forcing_ratio:
                decoder_input = target_seq[:, t, :].unsqueeze(1)
            else:
                decoder_input = decoder_output
        
        # Concatenate all outputs
        outputs = torch.cat(outputs, dim=1)
        return outputs