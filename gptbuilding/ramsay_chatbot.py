import torch, os, json, random, nltk
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

with open('intents.json', 'r') as f:
    text=f.read()

chars=sorted(list(set(text)))
vocab_size = len(chars)
stoi={ch:i for i, ch in enumerate(chars)}
itos={i:ch for i,ch in enumerate(chars)}
encode=lambda s:[stoi[c] for c in s]
decode=lambda l:"".join([itos[i] for i in l])
data=torch.tensor(encode(text),dtype=torch.long)
n=int(0.9*len(data))
traind,testd=data[:n],data[n:]
block_size=8

class Chatbot(torch.nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table=torch.nn.Embedding(vocab_size, vocab_size)


        self.first = torch.nn.Linear(input_size, 128)
        self.second = torch.nn.Linear(128, 64)
        self.third = torch.nn.Linear(64, output_size)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, idx,targets):

        logits=self.token_embedding_table(idx)

        #x = self.relu(self.first(x))
        #x = self.dropout(x)
        #x = self.relu(self.second(x))
        #x = self.dropout(x)
        #x = self.third(x)

        return logits



class ChatbotAssistant:
    def __init__(self, intents_path,block_size,batch_size, function_mappings=None):
        self.model = None
        self.intents_path = intents_path
        self.function_mappings = function_mappings
        self.documents = []
        self.vocabulary = []
        self.intents = []
        self.intents_responses = {}  # FIX 2: dict, not list
        self.X = None
        self.Y = None
        self.block_size = block_size
        self.batch_size=batch_size

    @staticmethod
    def tokenize_and_lemmatize(text):
        lemmatiser = nltk.WordNetLemmatizer()
        words = nltk.word_tokenize(text)
        words = [lemmatiser.lemmatize(word.lower()) for word in words]
        return words

    def bag_of_words(self, words):
        return [1 if word in words else 0 for word in self.vocabulary]  # FIX 3: self.vocabulary

    def parse_intents(self):
        if os.path.exists(self.intents_path):
            with open(self.intents_path, 'r') as f:
                intents_data = json.load(f)
            for intent in intents_data['intents']:
                if intent['tag'] not in self.intents:
                    self.intents.append(intent['tag'])
                    self.intents_responses[intent['tag']] = intent['responses']
                for pattern in intent['patterns']:  # FIX 1: moved inside the intent loop
                    p_words = self.tokenize_and_lemmatize(pattern)
                    self.vocabulary.extend(p_words)
                    self.documents.append((p_words, intent['tag']))
            self.vocabulary = sorted(set(self.vocabulary))

    def get_batch(self,split):
        data = traind if split == 'train' else testd
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack([data[i:i + self.block_size] for i in ix])
        y = torch.stack([data[i + 1:i + self.block_size + 1] for i in ix])
        return x,y

    def prepare_data(self):
        bags, indices = [], []
        for document in self.documents:
            words = document[0]
            bag = self.bag_of_words(words)
            intent_index = self.intents.index(document[1])
            bags.append(bag)
            indices.append(intent_index)
        self.X = np.array(bags)
        self.Y = np.array(indices)

    def train(self, batch_size, lr, epochs):
        X_tensor = torch.tensor(self.X, dtype=torch.float32)
        Y_tensor = torch.tensor(self.Y, dtype=torch.long)

        dataset = TensorDataset(X_tensor, Y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model = Chatbot(self.X.shape[1], len(self.intents))  # FIX 4: output size = number of intents

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            running_loss = 0.0
            for batch_X, batch_Y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_Y)
                loss.backward()
                optimizer.step()
                running_loss += loss
            print(running_loss / len(loader))

    def save_model(self, model_path, dimensions_path):
        torch.save(self.model.state_dict(), model_path)  # FIX 5a: dot, not comma
        with open(dimensions_path, 'w') as f:
            json.dump({'input_size': self.X.shape[1], 'output_size': len(self.intents)}, f)  # FIX 5b: colon

    def load_model(self, model_path, dimensions_path):
        with open(dimensions_path, 'r') as f:
            dimensions = json.load(f)
        self.model = Chatbot(dimensions['input_size'], dimensions['output_size'])
        self.model.load_state_dict(torch.load(model_path, weights_only=True))  # FIX 6: kwarg inside torch.load

    def process_message(self, message):
        words = self.tokenize_and_lemmatize(message)
        bag = self.bag_of_words(words)

        bag_tensor = torch.tensor(bag, dtype=torch.float32).unsqueeze(0)  # FIX 7: add batch dimension
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(bag_tensor)
        with torch.no_grad():
            predictions = self.model(bag_tensor)
        predicted_class_inx = torch.argmax(predictions, dim=1).item()  # FIX 7: .item() to get an int
        intent_p = self.intents[predicted_class_inx]
        if self.function_mappings:
            if intent_p in self.function_mappings:
                self.function_mappings[intent_p]()
        if self.intents_responses[intent_p]:
            return random.choice(self.intents_responses[intent_p])
        else:
            None





