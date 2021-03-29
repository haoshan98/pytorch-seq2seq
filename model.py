import random
import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, EMB_DIM, HID_DIM, EN_VOCAB_SIZE, NUM_LAYERS, DROPOUT, device):
        super(Encoder, self).__init__()
        self.device = device
        self.n_layers = NUM_LAYERS
        self.hidden_dim = HID_DIM
        self.emb = nn.Embedding(EN_VOCAB_SIZE, EMB_DIM)
        self.gru = nn.GRU(EMB_DIM, HID_DIM, NUM_LAYERS) 
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, sent):
        # print(f"en sent: {sent.shape}") # [seq len, batch size]
        x = self.dropout(self.emb(sent))
        # print(f"en emb: {x.shape}") # [seq len, batch size, emb dim]
        batch_size = sent.shape[1]
        # init_hidden = torch.zeros((batch_size, self.hidden_dim), device=self.device).unsqueeze(0)
        init_hidden = torch.zeros((self.n_layers, batch_size, self.hidden_dim), device=self.device)
        #hidden = [n layers * n directions, batch size, hid dim]
        out, hidden = self.gru(x, init_hidden)
        #outputs = [seq len, batch size, hid dim * n directions]
        # print(f"en hidden: {hidden.shape}")
        return F.relu(hidden)

# re-use the same context vector, 𝑧  returned by the encoder for every time-step in the decoder
# also pass the embedding of current token,  𝑑(𝑦𝑡)  and the context vector, 𝑧  to the linear layer.
class Decoder(nn.Module):
    def __init__(self, EMB_DIM, HID_DIM, DE_VOCAB_SIZE, NUM_LAYERS, DROPOUT):
        super(Decoder, self).__init__()

        self.emb = nn.Embedding(DE_VOCAB_SIZE, EMB_DIM)
        self.gru = nn.GRU(EMB_DIM + HID_DIM, HID_DIM, NUM_LAYERS) 
        self.linear = nn.Linear(HID_DIM *2 + EMB_DIM, DE_VOCAB_SIZE)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, sent, hidden, z):
        # (bs) -> (1, bs)
        sent = sent.unsqueeze(0)
        #input = [1, batch size]
        # print(f"de sent: {sent.shape}")
        emb = self.dropout(self.emb(sent))
        #embedded = [1, batch size, emb dim]
        # print(f"de emb: {emb.shape}")
        # emb_z_cat = torch.cat([emb.squeeze(0), z.squeeze(0)], dim=-1).unsqueeze(0)
        emb_z_cat = torch.cat([emb, z], dim=-1)

        out, hidden = self.gru(emb_z_cat, hidden)
        #output = [1, batch size, hid dim] 
        #hidden = z = [1, batch size, hid dim] 
        # print(f"de out: {out.shape}") # torch.Size([1, 4, 512])
        # print(f"de hidden: {hidden.shape}")
        # print(f"de z: {z.shape}")

        cat = torch.cat([out.squeeze(0), z.squeeze(0), emb.squeeze(0)], dim=-1) # cat on hid_dim & emb_dim
        # print(f"de cat: {cat.shape}") # torch.Size([4, 1280])
        out = self.linear(cat)
        # print(f"de hidden: {hidden.shape}")
        # print(f"de out: {out.shape}")
        return out, hidden


class Seq2seq(nn.Module):
    def __init__(self, EMB_DIM=256, HID_DIM=512, EN_VOCAB_SIZE=5893, DE_VOCAB_SIZE=7853, 
                NUM_LAYERS=1, ENC_DROPOUT=0.5, DEC_DROPOUT=0.5, device='cuda'):
        super(Seq2seq, self).__init__()
        self.device = device
        self.de_vocab_size = DE_VOCAB_SIZE
        self.encoder = Encoder(EMB_DIM, HID_DIM, EN_VOCAB_SIZE, NUM_LAYERS, ENC_DROPOUT, device)
        self.decoder = Decoder(EMB_DIM, HID_DIM, DE_VOCAB_SIZE, NUM_LAYERS, DEC_DROPOUT)

    def forward(self, en_sent, de_sent, teacher_forcing_ratio=0.5):
        hidden = self.encoder(en_sent)
        z = hidden
        max_length = de_sent.shape[0]
        batch_size = de_sent.shape[1]
        output = torch.zeros((max_length, batch_size, self.de_vocab_size), device=self.device)
        de_input = de_sent[0, :]
        # print(f"de_input <sos>: {de_input.shape}")
        for t in range(1, max_length):
            # print("t: ", t)
            out, hidden = self.decoder(de_input, hidden, z)
            output[t] =  out

            teacher_force = random.random() < teacher_forcing_ratio
            if teacher_force:
                de_input = de_sent[t, :]
                # print(f"next de_input force: {de_input.shape}")
            else:
                de_input = out.argmax(1)
                # print(f"next de_input argmax: {de_input.shape}")

        return output


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    en_sent = torch.randint(0, 5000, (15, 4)).to(device)
    de_sent = torch.randint(0, 6000, (14, 4)).to(device)

    model = Seq2seq().to(device)
    print(model)
    pred = model(en_sent, de_sent)
    print(pred.shape)
    # print(pred)

"""
Seq2seq(
  (encoder): Encoder(
    (emb): Embedding(5893, 256)
    (gru): GRU(256, 512)
    (dropout): Dropout(p=0.5, inplace=False)
  )
  (decoder): Decoder(
    (emb): Embedding(7853, 256)
    (gru): GRU(768, 512)
    (linear): Linear(in_features=1280, out_features=7853, bias=True)
    (dropout): Dropout(p=0.5, inplace=False)
  )
)
en sent: torch.Size([15, 4])
en emb: torch.Size([15, 4, 300])
en hidden: torch.Size([1, 4, 512])
de z: torch.Size([1, 4, 512])
de cat: torch.Size([4, 1280])
de_input <sos>: torch.Size([4])
t:  1
de sent: torch.Size([1, 4])
de emb: torch.Size([1, 4, 300])
de hidden: torch.Size([1, 4, 512])
de z: torch.Size([1, 4, 512])
de cat: torch.Size([4, 1280])
de out: torch.Size([1, 4, 7853])
next de_input force: torch.Size([4])
t:  2
de sent: torch.Size([1, 4])
de emb: torch.Size([1, 4, 300])
de hidden: torch.Size([1, 4, 512])
de z: torch.Size([1, 4, 512])
de cat: torch.Size([4, 1280])
de out: torch.Size([1, 4, 7853])
next de_input force: torch.Size([4])
t:  3
de sent: torch.Size([1, 4])
de emb: torch.Size([1, 4, 300])
de hidden: torch.Size([1, 4, 512])
de z: torch.Size([1, 4, 512])
de cat: torch.Size([4, 1280])
de out: torch.Size([1, 4, 7853])
next de_input argmax: torch.Size([1, 7853])
"""



