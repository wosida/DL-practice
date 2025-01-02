import torch
import torch.nn as nn
import torch.nn.functional as F
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(10, 12)
        self.encoder = nn.GRU(12,12,1,batch_first=True,bidirectional=False,bias=True)

    def forward(self, x):
        x = self.embedding(x)
        output, h_n = self.encoder(x)
        return output,h_n

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(10, 12)    #也有嵌入层，这个嵌入层是另一种语言的词汇，比如上面是中文，这里是英文
        self.decoder = nn.GRU(12,12,1,batch_first=True,bidirectional=False,bias=True)
        self.fc = nn.Linear(12,10) #out_features=num_embeddings=10,即词汇表大小（里面词语数量）
    def forward(self, x, h_n):
        x = self.embedding(x)
        x=F.relu(x)
        output, h_n = self.decoder(x,h_n)
        output = torch.sum(output,dim=1)
        output = self.fc(output)
        return output,h_n

class AttnDecoder(nn.Module):
    def __init__(self,hid_dim, output_dim, n_layers=1, dropout=0.1, max_len=10):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, hid_dim)
        self.rnn = nn.GRU(hid_dim, hid_dim, n_layers, batch_first=True, )
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.attn = nn.Linear(hid_dim * 2, max_len) #max_len是输入序列的长度
        self.attn_combine = nn.Linear(hid_dim * 2, hid_dim)
    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input)
        embedded = self.dropout(embedded)
        output, hidden = self.rnn(embedded, hidden)
        attn_weights = F.softmax(
            self.attn(torch.cat((output, hidden), 2)), dim=2)
        attn_applied = torch.bmm(attn_weights, encoder_outputs)
        output = torch.cat((output, attn_applied), 2)
        output = self.attn_combine(output).unsqueeze(1)
        output = F.relu(output)
        output = self.fc_out(output)
        return output, hidden, attn_weights

