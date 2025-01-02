import jieba
import torch
import torch.nn as nn
sentences=['明天去看展览','今天加班，天气不好','明天有图书展览','明天去']
labels=[1,0,1,1]
#分词，建立词表
sent_words=[]
vocab_dict={}
max_len=0
for sentence in sentences:
    words=list(jieba.cut(sentence))
    if max_len<len(words):
        max_len=len(words)
    sent_words.append(words)#二维列表
    #print(sent_words)  #[['明天', '去', '看', '展览'], ['今天', '加班', '，', '天气', '不好'], ['明天', '有', '图书', '展览'], ['明
    for word in words:
        vocab_dict[word]=vocab_dict.get(word,0)+1      #get() 是 Python 字典的方法，用来获取指定键（word）的值。如果键（word）存在于字典中，返回对应的值（即该单词之前出现过的次数）。如果键（word）不存在于字典中，返回默认值 0。
#print(vocab_dict)   #{'明天': 3, '去': 2, '看': 1, '展览': 2, '今天': 1, '加班': 1, '，': 1, '天气': 1, '不好': 1, '有': 1, '图书': 1}
#降序排列
sorted_vocab_dict=sorted(vocab_dict.items(),key=lambda x:x[1],reverse=True)    ## 使用 items() 获取字典中的所有键值对,元组的方式返回
#print(sorted_vocab_dict)   #[('明天', 3), ('去', 2), ('展览', 2), ('今天', 1), ('加班', 1), ('，', 1), ('天气', 1), ('不好', 1), ('有', 1), ('图书', 1)]
sorted_vocab_dict=sorted_vocab_dict[:-2]  #去掉最后两个
word2idx={'<unk>':0,'<pad>':1}#unk表示未知单词，pad表示填充
for word,_ in sorted_vocab_dict:
    word2idx[word]=len(word2idx)   #len(word2idx)是递增的,以此构建单词的唯一编码
#print(word2idx)  #{'<unk>': 0, '<pad>': 1, '明天': 2, '去': 3, '展览': 4}

#完成序列的索引编码，只有进行索引编码以后才可以张量化
max_len=int(max_len*0.9)
en_sentences=[]
for words in sent_words:
    words=words[:max_len]  #截取前max_len个单词
    #print(words) #['明天', '去', '看', '展览']
    en_words=[word2idx.get(word,0) for word in words]  #如果word在word2idx中，返回word2idx[word]，否则返回0
    #print(en_words) #[2, 3, 5, 6]
    en_words=en_words+[1]*(max_len-len(en_words))  #填充
    #print(en_words) #[2, 3, 5, 6, 1, 1, 1, 1]
    en_sentences.append(en_words) #二维列表
#print(en_sentences) #[[2, 3, 5, 6, 1, 1, 1, 1], [2, 3, 5, 6, 1, 1, 1, 1], [2, 3, 5, 6, 1, 1, 1, 1], [2, 3, 5, 6, 1, 1, 1, 1]]
#其实到这里，不过是把词换成了数字（索引），这个数字是按照词频排序的，所以词频高的数字小，词频低的数字大
#至此，每个序列形成等长的索引向量，保存在en_sentences中
#将en_sentences转化为张量
sentences_tensor=torch.LongTensor(en_sentences)
labels_tensor=torch.LongTensor(labels)
#print(sentences_tensor)
#print(labels_tensor)
#数据打包
dataset=torch.utils.data.TensorDataset(sentences_tensor,labels_tensor)
#print(dataset[0]) #元组，(tensor([2, 3, 5, 4]), tensor(1))
dataloder=torch.utils.data.DataLoader(dataset,batch_size=2,shuffle=True)#4个序列样本，每个俩个


class model(nn.Module):
    def __init__(self):
         super().__init__()
         self.embedding=nn.Embedding(len(word2idx),8)
         self.lstm=nn.LSTM(8,10,1,batch_first=True,bidirectional=False,bias=True)
         self.fc=nn.Linear(10,2)
    def forward(self,x):
        #print(x.shape)  #(B,L) 序列由单词索引组成
        x=self.embedding(x) #(B,L,8)  8是词向量维度，索引即是该矩阵的行索引
        #print(x.shape)
        x,(h,c)=self.lstm(x) #x:(B,L,10) h:(1,B,10) c:(1,B,10)
        x=torch.sum(x,dim=1)   #(B,10)
        x=self.fc(x)
        return x

m=model()
x=torch.randint(0,10,(2,4))
y=m(x)
print(y.shape)

