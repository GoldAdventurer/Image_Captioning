import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # The LSTM takes word embeddings as inputs, and 
        # outputs hidden states have a dimension equal to hidden_dim.
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, 
                            dropout=0.4 if num_layers > 1 else 0,
                            batch_first=True)
        #self.lstm = nn.LSTM(embed_size, hidden_size)
        # Add a linear layer that maps from hidden state space to vocab space
        #self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(hidden_size, vocab_size)
        #self.softmax = nn.LogSoftmax(dim=1)
        #self.hidden = self.init_hidden()

    #def init_hidden(self):
        # Initialize the hidden states
        #return (torch.zeros(1, 1, self.hidden_size))
        #        torch.zeros(1, 1, self.hidden_size))

    
    def forward(self, features, captions):
        #print('type(captions):', type(captions))
        captions = captions[:,:-1]
        #print('type(captions):', type(captions))        
        embeds = self.embedding(captions)
        #embeds = F.relu(embed)
        embeds = torch.cat((features.unsqueeze(1), embeds), 1)
        outputs, hiddens = self.lstm(embeds)
        #outputs = self.dropout(outputs)   
        outputs = self.linear(outputs) 
        return outputs      


    def sample(self, inputs, states=None, max_len=20):
            #" accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        res = []
        #prev_state = None

        for i in range(max_len):
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens)
            _, predicted = torch.max(outputs,2)
            inputs = self.embedding(predicted)
            predicted_idx = predicted.item()
            res.append(predicted_idx)
            # if the predicted idx is the stop index, the loop stops
            if predicted_idx == 1:
                break
        #res = torch.cat(res, 1)
        #res = res.cpu().data.numpy()
        #return res.tolist()[0]
        return res
