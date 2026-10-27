import json,re,requests
import numpy as np
import torch, random
import torch.nn.functional as F
import torch.optim as optim
#url="https://en.wikiquote.org/w/api.php?action=query&prop=revisions&rvprop=content&rvslots=main&format=json&titles=Kitchen_Nightmares_(uncensored)"
#headers = {"User-Agent": "RamsayBot/1.0 (wickedposh@gmail.com) learning project"}
#response=requests.get(url,headers=headers)
#data=response.json()
#pages=data["query"]["pages"]
#pid=next(iter(pages))
#wikitext = pages[pid]['revisions'][0]['slots']['main']['*']
#text = re.sub(r'{{[^']+}}', '',wikitext )
#text = re.sub(r'==[^']+==' , '',text )
#text = re.sub(r'\[[^\]]*\]', '', text)
#text = re.sub(r'\([^)]*\)', '', text)
#text = re.sub(r'<[^>]+>','',text)
#pattern = r":'''([^']+)''':\s*(.*)"
#lines=[]
#for speaker, line in re.findall(pattern, text):
#    line = line.strip()
#    if line:
#        lines.append(f":{speaker}: {line}")
#output = "\n".join(lines)
#with open("ramsey.txt", "w", encoding="utf-8") as f:
#    f.write(output)
with open('intent.json', 'r') as f:
    datas = json.loads(f.read())
words,documents,classes=[],[],[]

def clean_and_tokenize(text):
    lower=text.lower()
    tokens = re.sub(r'[^a-z0-9\s]', '', lower).split()
    return tokens
for intent in datas["intents"]:
    for pattern in intent["patterns"]:
        w=clean_and_tokenize(pattern)
        documents.append((w,intent['tag']))
        words.extend(w)
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
words=sorted(list(set(words)))
classes=sorted(list(set(classes)))

def vectorize(tokens):
    vec=[tokens.count(w) for w in words]
    return np.array(vec)

def calculate(vec1,vec2):
    if np.linalg.norm(vec1)==0 or np.linalg.norm(vec2)==0:
        return 0
    cosine=vec1.dot(vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))
    return cosine
doc_vectors=[(vectorize(tokens),tag) for tokens,tag in documents]
def score(u):
    best_sim=0
    best_tag=''
    for vec, tag in doc_vectors:
        sim=calculate(u,vec)
        if sim>best_sim:
            best_sim,best_tag=sim,tag
    return best_sim, best_tag
responses = {intent["tag"]: intent["response"] for intent in datas["intents"]}   # build once, up top
threshold = 0.5

def router(user_input):
    u=vectorize(clean_and_tokenize(user_input))
    sim,tag=score(u)
    if sim >= threshold:
        r=responses[tag]
        return random.choice(r) if isinstance(r,list) else r
    else:
        return best.chat(user_input,role=":Gordon:")

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
for i in range(100):
    stats=get_stats(ids)
    pair=max(stats,key=stats.get)
    idx=256+i
    ids=merge(ids,pair,idx)
    merges[pair]=idx
vocab={idx:bytes([idx]) for idx in range(256)}
for (pair,idx) in merges.items():
    vocab[idx]=vocab[pair[0]]+vocab[pair[1]]
vocab_size=256+len(merges)

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
        self.history = ""

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
    @torch.no_grad()
    def chat(self, user_text, role=":Gordon:"):
          self.history += f":User:{user_text}\n"
          prompt = self.history + f"{role}"
          ids = torch.tensor([encode(prompt)], dtype=torch.long)
          output = self.model.generate(ids, max_new_tokens=50)
          text = decode(output[0][ids.shape[1]:].tolist())
          response = text                      # default: whole reply
          stop = re.search("\n:[^:\n]+:",text)
          if stop:
              response = response[:stop.start()]
          self.history += f"{role}{response.strip()}\n"
          self.history = self.history[-1000:]
          return response.strip()

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



best=RamseyAssistant(32, 32, 3000,0.01,vocab_size)
best.train('train')
while True:
    user_input = input("You:")
    if user_input.lower() == "quit":
        break
    reply = router(user_input)
    print(":Gordon:", reply)