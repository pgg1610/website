---
date: "2020-12-18"
description: Simple uncertainity estimate using dropout in PyTorch. 
pubtype: Jupyter Notebook
fact: 
featured: true
image: /img/simple_dropout/final_image.png
link: https://github.com/pgg1610/misc_notebooks/blob/master/Simple_dropout.ipynb
tags:
- Python
- PyTorch
title: Simple Dropout using PyTorch
---

* Adapted from Deep Learning online course notes from NYU. [Note link](https://atcold.github.io/pytorch-Deep-Learning/en/week14/14-3/)
* [Paper about using Dropout as a Bayesian Approximation](https://arxiv.org/pdf/1506.02142.pdf)

In addition to predicting a value from a model it is also important to know the confidence in that prediction. Dropout is one way of estimating this. After multiple rounds of predictions, the mean and standard deviation in the prediction can be viewed as the prediction value and the corresponding confidence in the prediction. It is important to note that this is different from the error in the prediction. The model may have error in the prediction but could be precise in that value. It is similar to the idea of accuracy vs precision. 

When done with dropout -- the weights in the NN are scale by $\frac{1}{1-r}$ to account for dropping of the weights 

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

Once a NN with a dropout implemented is instantiated, the model is called multiple times to predict the output for a given input. While doing so it is important to ensure the model is in `train()` state. 

```python
def predict_reg(model, X, T=1000):
    
    model = model.train()
    Y_hat = list()
    with torch.no_grad():
        for t in range(T):
            Y_hat.append(model(X.view(-1,1)).squeeze())
    Y_hat = torch.stack(Y_hat)
    
    model = model.eval()
    with torch.no_grad():
        Y_eval = model(X.view(-1,1)).squeeze()

    return Y_hat, Y_eval
```

 
![final](/img/simple_dropout/final_image.png)
