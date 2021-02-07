---
date: "2021-02-01"
description: Simple example of graph neural network
pubtype: Jupyter Notebook
fact: 
featured: true
image: /img/Simple_GCN/springlayout.png
link: https://github.com/pgg1610/misc_notebooks/blob/master/Bayesian_optimisation/modular/demo.ipynb
tags:
- Python
- Pytorch
- Graphs
title: Simple Graph Neural Network
---


Simple example to illustrate the utility of graph neural networks. 

**Task:** Generating embedding for a graph dataset using a Graph Convolution Neural Network (GCN) on Zachary's Karate Club Network. Categorize the members of the club 

* Data file from: [Zachary W. (1977). An information flow model for conflict and fission in small groups. Journal of Anthropological Research, 33, 452-473](http://vlado.fmf.uni-lj.si/pub/networks/data/Ucinet/UciData.htm)

This is a classic dataset to look at relationships between users and its final effect on the decision. The dataset describes the social interaction of 34 members and the communities that rise from it. The new version of the data categorizes the nodes (each member) in 4 clubs. These clubs are emerged from the connections each members has with others in the 'club'. 

### How is this useful? 
- Calculate embedding to compress the graph dataset into 2 dimensions 
- Can we predict communities of club members based on their vicinity with other members 

## Setup 


Karate Club Dataset:

======================

Number of graphs: 1

Number of features: 34

Number of classes: 4

![spring_layout](/img/Simple_GCN/springlayout.png)
	
Each node in the graph is a person. Every person has an associated number (index) and the club they would eventually join. In this form of visualization - node 0 and node 33 are Mr. Hi and Officer respectively. Besides that, each node has an associated edges with other nodes in the network based on connections (how exactly are those determined is not clear at first). Now having that connection we can construct an adjacency matrix. The environment of each node can be used to predict the final community the user would end up in. 

We can re-express this problem as given the nodes and the connections which club would each node join. We can see if the GCN network can predict the targets properly or rather if the targets can be used to find low dimensional embeddings for the graph objects such that each node is expressed in 2D. 

![barplot](/img/Simple_GCN/barplot_classes.png)

To describe each members in the network a one-hot encoding is used where the entry corresponding to the index of the node is 1 and everything else is 0. Sorting these nodes based on index we get a identity matrix (34, 34)

```
tensor([[1., 0., 0.,  ..., 0., 0., 0.],
        [0., 1., 0.,  ..., 0., 0., 0.],
        [0., 0., 1.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 1., 0., 0.],
        [0., 0., 0.,  ..., 0., 1., 0.],
        [0., 0., 0.,  ..., 0., 0., 1.]])
```

Next, every node in the graph is attached to other nodes. This information is stored in adjacency matrix. Self-connections are by default labelled 0. Every row of the adjacency matrix shows node connections

```
tensor([[0., 1., 1.,  ..., 1., 0., 0.],
        [1., 0., 1.,  ..., 0., 0., 0.],
        [1., 1., 0.,  ..., 0., 1., 0.],
        ...,
        [1., 0., 0.,  ..., 0., 1., 1.],
        [0., 0., 1.,  ..., 1., 0., 1.],
        [0., 0., 0.,  ..., 1., 1., 0.]])
```

For example: as shown figure at the top if we look at node (16) it is connected to node (5,6) only. Hence in the adjacency matrix those index are 1 other that all other entries are 0

```python
np.where( Karate_adjacency[16] == 1 )[0]
>> array([5, 6])
```

**Utility of graph neural network:**

In this example we test the application of graph neural network to cluster/identify communities in the graph given information about the connections of the members and data of only 4 members' community choice. Given the 4 datapoints our GNN would predict and subsequently cluster the other members of the network

*1. Tipf's Graph Convolution Implementation*

```python
class GCNConv(nn.Module):
    def __init__(self, A, input_dims, output_dims):
        super(GCNConv, self).__init__()
        '''
        As per Tipf explanation: 
        https://tkipf.github.io/graph-convolutional-networks/
        https://arxiv.org/abs/1609.02907
        
        PARAMETERS: 
        ---------------
        A: numpy.array, Adjacency matrix for the graph object 
        input_dims: int, Input dimensions for the NN params
        output_dims: int, Output dimensions for the NN params 
        
        RETURNS: 
        ---------------
        out: torch.Tensor, N x output for the NN prediction
        '''
        torch.manual_seed(42)
        
        self.A_hat = A + torch.eye(A.size(0))
        self.D     = torch.diag(torch.sum(A,1)) #Diagonal node-degree matrix 
        self.D     = self.D.inverse().sqrt()
        self.A_hat = torch.mm( torch.mm(self.D, self.A_hat), self.D )
        self.W     = nn.Parameter(torch.rand(input_dims, output_dims, requires_grad=True))
    
    def forward(self, X):
        out = torch.tanh(torch.mm( torch.mm(self.A_hat, X), self.W ))
        
        return out
```

This module is used when making the neural network modeling 

```python 
class Net(torch.nn.Module):
    def __init__(self, A, nfeat, nhid, c):
        super(Net, self).__init__()
        self.conv1 = GCNConv(A, nfeat, nhid)
        self.conv2 = GCNConv(A, nhid, nhid)
        self.conv3 = GCNConv(A, nhid, 2)
        self.linear = nn.Linear(2, nhid)
        
    def forward(self,X):
        H0  = self.conv1(X)
        H1 = self.conv2(H0)
        H2 = self.conv3(H1)
        out = self.linear(H2)
        
        return H2, out 
```

### Training the model 

```python 
for i in range(1000):
    e, out = simple_GCN(node_features)
    optimizer.zero_grad()
    loss=criterion(out[data.train_mask], data.y[data.train_mask])
    #loss=criterion(out, data.y)
    loss.backward()
    optimizer.step()
    if i % 100==0:
        print("Step: {} Cross Entropy Loss = {}".format(i, loss.item()))
        output_, _ = simple_GCN(node_features)
        visualize_graph(data, output_)
```

First forward-pass result:

![first_simple](/img/Simple_GCN/simple/first.png)

Final optimized weights prediction: 

![final_simple](/img/Simple_GCN/simple/final_output.png)


*2. PyTorch Geometric Implementation*

This part is adapted from PyTorch Geometric's tutorial page. [Link](https://colab.research.google.com/drive/1h3-vJGRVloF5zStxL5I0rSy4ZUPNsjy8?usp=sharing#scrollTo=7cjjyFVnpKB0)

```python
from torch_geometric.nn import GCNConv
from torch_geometric.utils import add_self_loops, degree

class GCN(torch.nn.Module):
    def __init__(self, graph_data):
        super(GCN, self).__init__()
        torch.manual_seed(42)
        self.graph_data = graph_data
        self.conv1 = GCNConv(self.graph_data.num_features, 4)
        self.conv2 = GCNConv(4, 4)
        self.conv3 = GCNConv(4, 2)
        self.classifier = nn.Linear(2, self.graph_data.num_classes)

    def forward(self, node_features, edge_index):
        h = self.conv1(node_features, edge_index)
        h = h.tanh()
        h = self.conv2(h, edge_index)
        h = h.tanh()
        h = self.conv3(h, edge_index)
        h = h.tanh()  # Final GNN embedding space.
        
        # Apply a final (linear) classifier.
        out = self.classifier(h)

        return out, h
```

### Training the model 

```python 
criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Define optimizer.

def train(data):
    optimizer.zero_grad()  # Clear gradients.
    out, h = model(node_features, data.edge_index)  # Perform a single forward pass.
    loss = criterion(out[data.train_mask], data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    return loss, h

for epoch in range(1000):
    loss, h = train(data)
    if epoch % 100 == 0:
        print("Step: {} Cross Entropy Loss = {}".format(epoch, loss.item()))
        visualize_graph(data, h)
```

First forward-pass result:

![pyg_first](/img/Simple_GCN/pyg_gcn/first.png)

Final optimized weights prediction: 

![pyg_final](/img/Simple_GCN/pyg_gcn/final_output.png)

