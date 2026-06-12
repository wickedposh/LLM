import torch, os, random
import torch.nn.functional as F
import torch.optim as optim

with open('ramsey.txt', 'r') as f:
    text = f.read()

torch.manual_seed(324)
tokens=text.encode('utf-8')
tokens=list(map(int,tokens))
def get_stats(ids):
    counts={}
    for pair in zip(ids,ids[1:]):
        counts[pair]=counts.get(pair,0)+1
    return counts
def merge(ids,pair,idx):
    newids=[]
    i=0
    while i<len(ids):
        if i<len(ids)-1 and ids[i+1]==pair[1] and ids[i]==pair[0]:
            newids.append(idx)
            i+=2
        else:
            newids.append(ids[i])
            i+=1
    return newids

ids=list(tokens)
merges={}
for i in range(1000):
    stats=get_stats(ids)
    pair=max(stats,key=stats.get)
    idx=256+i
    ids=merge(ids,pair,idx)
    merges[pair]=idx
vocab={idx:bytes([idx]) for idx in range(256)}
for (pair,idx) in merges.items():
    vocab[idx]=vocab[pair[0]]+vocab[pair[1]]


def decode(ids):
    tokens=b"".join(vocab[idx] for idx in ids)
    text=tokens.decode('utf-8',errors="replace")
    return text


def encode(text):
    tokens=list(text.encode('utf-8'))
    while len(tokens)>=2:
        stats=get_stats(tokens)
        pair=min(stats,key=lambda p:merges.get(p,float("inf")))
        if pair not in merges:
            break
        idx=merges[pair]
        tokens=merge(tokens,pair,idx)
    return tokens
vocab_size=256+len(merges)

#chars = sorted(list(set(text)))
#vocab_size = len(chars)
#stoi = {ch: i for i, ch in enumerate(chars)}
#itos = {i: ch for i, ch in enumerate(chars)}
#encode = lambda s: [stoi[c] for c in s]
#decode = lambda l: "".join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
traind, testd = data[:n], data[n:]
block_size = 32
n_embd=64
dropout=0.2
n_heads=4
n_layer=3

def get_batch(split, batch_size,block_size):
    data = traind if split == 'train' else testd
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y

class Ramsey(torch.nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = torch.nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = torch.nn.Embedding(block_size, n_embd)
        self.blocks=torch.nn.Sequential(*[Block(n_embd,n_heads=n_heads) for _ in range(n_layer)])
        self.ln_f=torch.nn.LayerNorm(n_embd)
        self.lm_head=torch.nn.Linear(n_embd,vocab_size)
        self.vocab_size = vocab_size


    def forward(self, idx, targets=None):
        B,T=idx.shape
        tok_emb=self.token_embedding_table(idx)
        pos_emb=self.position_embedding_table(torch.arange(T))
        x=tok_emb+pos_emb
        x=self.blocks(x)
        x=self.ln_f(x)
        logits=self.lm_head(x)


        if targets is not None:
            B,T,C=logits.shape
            logits=logits.view(B*T,C)
            targets=targets.view(B*T)
            loss=F.cross_entropy(logits, targets)
        else:
            loss=None

        return logits,loss

    def generate(self,idx,max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond=idx[:,-block_size:]
            logits,loss=self(idx_cond)
            logits=logits[:,-1,:]
            probs=F.softmax(logits,dim=-1)
            idx_next=torch.multinomial(probs,1)
            idx=torch.cat((idx,idx_next),dim=1)
        return idx



class RamseyAssistant:
    def __init__(self, block_size, batch_size, epochs,lr,vocab_size):
        self.vocab_size = vocab_size
        self.model = Ramsey(self.vocab_size)
        self.block_size = block_size
        self.batch_size = batch_size
        self.epochs=epochs
        self.lr = lr
        self.vocab_size = self.vocab_size

    def train(self, x):
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)
        for steps in range(self.epochs):
            xb, yb = get_batch(x,self.batch_size,self.block_size)
            logits, loss = self.model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if steps % 100 == 0:
                print(f"  lr={self.lr} step={steps} loss={loss.item():.3f}")

    @torch.no_grad()
    def estimate_loss(self,split):
        out={}
        self.model.eval()
        for split in ['train', 'test']:
            losses=torch.zeros(self.batch_size)
            for k in range(self.batch_size):
                X,Y=get_batch(split,self.batch_size,self.block_size)
                logits,loss=self.model(X,Y)
                losses[k]=loss.item()
            out[split]=losses.mean()
        self.model.train()
        return out


class Head(torch.nn.Module):
    def __init__(self,head_size):
        super().__init__()
        self.key=torch.nn.Linear(n_embd,head_size,bias=False)
        self.query=torch.nn.Linear(n_embd,head_size,bias=False)
        self.value=torch.nn.Linear(n_embd,head_size,bias=False)
        self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size)))
        self.dropout=torch.nn.Dropout(dropout)
        self.head_size=head_size
    def forward(self,x):
        B,T,C=x.shape
        k=self.key(x)
        q=self.query(x)
        wei=q@k.transpose(-2,-1)*self.head_size**-0.5
        wei=wei.masked_fill(self.tril[:T,:T]==0,float('-inf'))
        wei=F.softmax(wei,dim=-1)
        wei=self.dropout(wei)
        v=self.value(x)
        out=wei@v
        return out

class MultiHeadAttention(torch.nn.Module):
    def __init__(self,n_heads,head_size):
        super().__init__()
        self.heads=torch.nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        self.proj=torch.nn.Linear(n_embd,n_embd)
        self.dropout=torch.nn.Dropout(dropout)
    def forward(self,x):
        out=torch.cat([h(x) for h in self.heads],dim=-1)
        out=self.proj(out)
        return self.dropout(out)

class FeedForward(torch.nn.Module):
    def __init__(self,n_embd):
        super().__init__()
        self.net=torch.nn.Sequential(torch.nn.Linear(n_embd,4*n_embd),
                                     torch.nn.ReLU(),
                                     torch.nn.Linear(4*n_embd,n_embd),
                                     torch.nn.Dropout(dropout),)
    def forward(self,x):
        return self.net(x)

class Block(torch.nn.Module):
    def __init__(self,n_embd,n_heads):
        super().__init__()
        head_size=n_embd//n_heads
        self.sa=MultiHeadAttention(n_heads,head_size)
        self.ffwd=FeedForward(n_embd)
        self.ln1=torch.nn.LayerNorm(n_embd) ##normalisation
        self.ln2=torch.nn.LayerNorm(n_embd)
    def forward(self,x):
        x=self.sa(self.ln1(x))+x ##residual connection
        x=self.ffwd(self.ln2(x))+x
        return x

learning_rate=[0.1,0.01,0.001,0.0001]
result=[]
for i in range(len(learning_rate)):
    m=RamseyAssistant(32, 16, 500,learning_rate[i],vocab_size)
    m.train(traind)
    a=m.estimate_loss('train')
    result.append((i,a['train'],a['test']))
print(result)

context=torch.zeros((1,1),dtype=torch.long)
best=RamseyAssistant(32, 16, 3000,0.01,vocab_size)
best.train(traind)
print(decode(best.model.generate(context,max_new_tokens=1000)[0].tolist()))
