import torch
import torch.nn as nn
from torchvision.models import vgg19


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.net = vgg19(pretrained=True)
        self.net = nn.Sequential(*list(self.net.features.children())[:-1])

    def forward(self, x):
        x = self.net(x)
        x = x.view(x.size(0), x.size(1), -1).transpose(1,2)
        return x

class Attention(nn.Module):
    def __init__(self, a_dim, h_dim, attn_dim):
        super(Attention, self).__init__()
        self.linear_a = nn.Linear(a_dim, attn_dim)
        self.linear_h = nn.Linear(h_dim, attn_dim)
        self.f_attn = nn.Linear(attn_dim, 1)

    def forward(self, a, h):
        # param a : encoder output (a tensor with size [batch, L, D])
        # param h : decoder hidden state (a tensor with size [batch, h_dim])
        a = self.linear_a(a) # [batch, L, attn_dim]
        h = self.linear_h(h).unsqueeze(1) # [batch, 1, attn_dim]

        e = self.f_attn(F.relu(a + h)) # h will be broadcasted. [batch, L, 1]
        e = e.squeeze(-1) # [batch, L]

        alpha = F.softmax(e, dim = -1) # [batch, L]
        z = torch.sum(a * alpha.unsqueeze(2), dim = 1) # [batch, D]

        return z, alpha

class Decoder(nn.Module):
    def __init__(self, a_dim, h_dim, attn_dim, vocab_size, embed_dim):
        super(Decoder, self).__init__()
        self.LSTM = nn.LSTMCell(a_dim + embed_dim, h_dim)
        self.attention = Attention(a_dim, h_dim, attn_dim)
        self.f_init_c = nn.Linear(a_dim, h_dim)
        self.f_init_h = nn.Linear(a_dim, h_dim)
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Deep output layer
        self.f_out = nn.Linear(embed_dim, vocab_size, bias = False)
        self.f_h = nn.Linear(h_dim, embed_dim, bias = False)
        self.f_z = nn.Linear(a_dim, embed_dim, bias = False)

        # doubly stochastic attention
        self.f_beta = nn.Linear(h_dim, a_dim)

    def forward(self, a, trg):
        # param a : encoder output (a tensor with size [batch, L, a_dim])
        # param trg : target caption (a tensor with size [batch, C])
        max_caption_len = trg.size(1)
        pred, alpha = None, None

        h_t, c_t = self.f_init_h(torch.mean(a, dim = 1)), self.f_init_c(torch.mean(a, dim = 1))
        z_t = self.attention(a, h_t) # [batch, a_dim]
        y_t = trg[:,0:1] # [batch, 1]

        for t in range(max_caption_len):
            # main procedure
            Ey_t = self.embedding(y_t) # [batch, 1, embed_dim]
            beta = F.sigmoid(self.f_beta(h_t)) # [batch, h_dim] -> [batch, a_dim]
            
            lstm_input = torch.cat([Ey_t, z_t.unsqueeze(1)], dim = -1) # [batch, 1, a_dim + embed_dim]
            h_t, c_t = self.LSTM(lstm_input, (h_t, c_t)) # [batch, 1, h_dim], [batch, 1, h_dim]
            z_t, alpha_t = self.attention(a, h_t) # [batch, a_dim], [batch, L]

            # doubly stochastic attention
            z_t = beta * z_t
            
            # Deep output layer
            output = self.f_out(Ey_t.squeeze(1) + self.f_h(h_t.squeeze(1)) + self.f_z(z_t)) # [batch, vocab_size]
            
            #prob = F.softmax(output, dim = -1) # [batch, vocab_size]
            prob = output # exclude softmax to use CrossEntropyLoss

            if pred is None:
                pred = prob.unsqueeze(1) # [batch, 1, vocab_size]
                alpha = alpha_t.unsqueeze(1) # [batch, 1, a_dim]
            else:
                pred = torch.cat([pred, prob.unsqueeze(1)], dim = 1) # [batch, t+1, vocab_size]
                alpha = torch.cat([alpha, alpha_t.unsqueeze(1)], dim = 1) # [batch, t+1, a_dim]
            
            # next word
            if t < max_caption_len - 1:
                y_t = trg[:,t+1:t+2] # teacher forcing
        
        return pred, alpha # [batch, C, vocab_size], [batch, C, L]