import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        self.batch_norm = nn.BatchNorm1d(embed_size,momentum=0.01)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        features = self.batch_norm(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN,self).__init__()
        
        self.hidden_size = hidden_size
        
        self.word_embeddings = nn.Embedding(vocab_size,embed_size)
        self.lstm = nn.LSTM(embed_size,hidden_size,num_layers=1,batch_first=True)
        self.linear_layer = nn.Linear(hidden_size,vocab_size)
        
    def forward(self, features, captions):
        
        # we are not decoding the final <end> token
        embeds = self.word_embeddings(captions[:,:-1])
        inputs = torch.cat((features.unsqueeze(1),embeds),1)
  
        lstm_out,_ = self.lstm(inputs)
    
        out = self.linear_layer(lstm_out)
        
        return out
    

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        tensor_ids = []
        
        for ten_id in range(max_len):
            
            lstm_out,states = self.lstm(inputs,states)                            # 3D tensor for lstm (batch_size,1,embed_size)
            outputs = self.linear_layer(lstm_out.squeeze(1))                      # (batch_size,hidden_size)
            max_val,idx = outputs.max(1)
            tensor_ids.append(int(idx[0]))
            inputs = self.word_embeddings(idx)
            inputs = inputs.unsqueeze(1)
        
        return tensor_ids
        
        
        pass