---
date: "2020-12-18"
description: Simple uncertainity estimate using dropout in PyTorch. 
pubtype: Jupyter Notebook
fact: 
featured: true
image: /img/food_graph/frequency_plot.png
link: https://github.com/pgg1610/misc_notebooks/blob/master/Simple_dropout.ipynb
tags:
- Python
- PyTorch
title: Simple Dropout using PyTorch
---

```python 
# Define a simple NN 
class MLP(nn.Module):
    def __init__(self, hidden_layers=[20, 20], droprate=0.2, activation='relu'):
        super(MLP, self).__init__()
        
        self.model = nn.Sequential()
        self.model.add_module('input', nn.Linear(1, hidden_layers[0]))
        
        if activation == 'relu':
            self.model.add_module('relu0', nn.ReLU())
        
        elif activation == 'tanh':
            self.model.add_module('tanh0', nn.Tanh())
            
        for i in range(len(hidden_layers)-1):
            self.model.add_module('dropout'+str(i+1), nn.Dropout(p=droprate))
            self.model.add_module('hidden'+str(i+1), nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            
            if activation == 'relu':
                self.model.add_module('relu'+str(i+1), nn.ReLU())
                
            elif activation == 'tanh':
                self.model.add_module('tanh'+str(i+1), nn.Tanh())
                
        self.model.add_module('dropout'+str(i+2), nn.Dropout(p=droprate))
        self.model.add_module('final', nn.Linear(hidden_layers[i+1], 1))
        
    def forward(self, x):
        return self.model(x)
```
![final](/img/simple_dropout/final_image.png)
