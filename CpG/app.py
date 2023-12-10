import numpy as np
from flask import Flask, request,render_template
print('testing')
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from typing import Sequence
from functools import partial
from torch.utils.data import Dataset, DataLoader
print('testing')
import pickle
import torch 
from functools import partial
app = Flask(__name__)


# Regression Model
class LSTM(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size,
            num_layers=num_layers
        )
        
        self.fc = nn.Linear(hidden_size,self.num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(1), self.hidden_size))
        
        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(1), self.hidden_size))
        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))
        
        out = self.fc(h_out)
        out = self.relu(out)
        
        return out

input_size = 128
max_len = 128 
hidden_size = 64
num_layers = 1
num_classes = 1
batch_size = 16
SAVED_MODEL = LSTM(num_classes, input_size, hidden_size, num_layers)
SAVED_MODEL.load_state_dict(torch.load('saved_model.pth'))

@app.route('/',methods = ['GET'])
def home():
    return render_template('index.html')


@app.route("/predict",methods=['POST'])
def predict():
    if request.method == 'POST':
        Medu = request.form.get("Medu")	
        # find the output using model
        output = None 
        def single_predict(test_sample):

            alphabet = 'NACGT'
            dna2int = {a: i for a, i in zip(alphabet, range(1, 6))}
            int2dna = {i: a for a, i in zip(alphabet, range(1, 6))}
            dna2int.update({"pad": 0})
            int2dna.update({0: "<pad>"})

            test_sample = list(test_sample)
            print(test_sample)
            test_sample = [dna2int[i] for i in test_sample]
            test_sample = np.array(test_sample)
            print(test_sample) 
            with torch.inference_mode():
                observation = torch.from_numpy(np.array(test_sample)).float()
                temp_obs = torch.zeros(max_len-observation.shape[0])
                observation = torch.cat((observation,temp_obs),dim=0)
                observation = observation.unsqueeze(0).unsqueeze(0)
                observation_output = SAVED_MODEL(observation).item()
                return observation_output
        
        output = single_predict(Medu)
        return render_template('index.html', prediction_text= "Lstm Model Prediction is {}".format(output))
    else:
        return render_template('index.html')

    
if __name__ == "__main__":
    app.run(debug=True)
    

