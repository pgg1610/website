---
date: "2021-02-01"
description: Simple example of applying graph neural networks for node classification
pubtype: Jupyter Notebook
fact: 
featured: true
image: /img/Simple_GCN/springlayout.png
link: https://github.com/pgg1610/misc_notebooks/blob/master/graphs/GCN_basics_w_pytorch_geometric.ipynb
tags:
- Python
- Pytorch
- Graphs
title: Simple Graph Neural Network
---


Simple example to illustrate the utility and working of graph neural networks. 


A natural way to represent information in a structured form is as a graph. A graph is a data structure describing a collection of entities, represented as nodes, and their pairwise relationships, represented as edges. Think of it as a mathematical abstraction to present relational data. 

Graphs are everywhere: social networks, the world wide web, street maps, knowledge bases used in search engines, and even chemical molecules are frequently represented as a set of entities and relations between them. 

Machine learning deals with the question of how we can build systems and design algorithms that learn from data and experience (e.g., by interacting with an environment), which is in contrast to the traditional approach in science where systems are explicitly programmed to follow a precisely outlined sequence of instructions. The problem of learning is commonly approached by fitting a model to data with the goal that this learned model will generalize to new data or experiences.

Furthermore, Graph network learning provides a promising combination of two ideas: 

(1) having strong relational inductive bias for a data structure which is amenable for graph representation 

(2) find hidden features/reprenstation that can be 'learned' with more data. 

This idea is explored in futher details in this fairly exhaustive [review of graph networks](https://arxiv.org/abs/1806.01261)

----------------

In this post we will look at a simple case of using Graph Neural Networks to aid labeling and separating nodes in the graph structured data. 

**Task:** Generating embedding for a graph dataset using a Graph Convolution Neural Network (GCN) on Zachary's Karate Club Network. 

* Dataset: [Zachary W. (1977). An information flow model for conflict and fission in small groups. Journal of Anthropological Research, 33, 452-473](http://vlado.fmf.uni-lj.si/pub/networks/data/Ucinet/UciData.htm)

The dataset describes the social interaction of 34 members and the communities that rise from it, 4 in this case. Each members of the club is defined as a node. Each node is connected to other members in the club. This connection would determine the final grouping of the community in 4 separate labels.

### How is GNN useful here? 
Consider a situation where: We knew how every one is connected to each other in the club. However we only know the final label of only 4 members in the club. That means, out of 34 members we know only 4 members' final label. Can we use the node connections and the GCN idea to predict and cluster other members of the group? In addition currently the data is structured in a tuple of (node, edge connections) can we use graph neural network to estimate lower dimensional embedding to describe each node? 

## Setup 


Karate Club Dataset:

======================

Number of graphs: 1

Number of features: 34

Number of classes: 4

```python
from torch_geometric.utils import to_networkx
from torch_geometric.datasets import KarateClub

dataset = KarateClub()
G = to_networkx(dataset[0], to_undirected=True)
fig, ax = plt.subplots(1,1,figsize=(10,10))

#Plot the dataset using Networkx
nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), node_size=10**3, with_labels=True, node_color=data.y, cmap="Set2", ax=ax)
```

![spring_layout](/img/Simple_GCN/springlayout.png)
	
Like introduced in the previous section: Each node in the graph is a person. Every person has an associated number (index) and the community or club they would eventually join. There are 4 clubs in total. Each node has an associated edges with other nodes in the network based on connections. Now having that connection we can construct an adjacency matrix. The environment of each node can be used to predict the final community the user would end up in. 

We can re-express this problem as given the nodes and the connections which club would each node join. We can see if the GCN network can predict the targets properly, and if the targets can be used to find low dimensional representation for the graph.

To describe each members in the network a one-hot encoding is used where the entry corresponding to the index of the node is 1 and everything else is 0. Sorting these nodes based on index we get a identity matrix (34, 34). More elaborate schemes can be thought of to describe the node entries. Like in case of molecule property prediction each atom which would be a node can be expressed as combination of chemical properties. 

**Node features used to describe the 34 members:**
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

**Node adjacency matrix used to describe the connections of 34 nodes:**
```
tensor([[0., 1., 1.,  ..., 1., 0., 0.],
        [1., 0., 1.,  ..., 0., 0., 0.],
        [1., 1., 0.,  ..., 0., 1., 0.],
        ...,
        [1., 0., 0.,  ..., 0., 1., 1.],
        [0., 0., 1.,  ..., 1., 0., 1.],
        [0., 0., 0.,  ..., 1., 1., 0.]])
```

For example: as shown figure at the top if we look at node (16) it is connected to node (5,6). Hence in the adjacency matrix the entries belonging to node (16) are index (5,6) are 1, other that all other entries are 0

```python
np.where( Karate_adjacency[16] == 1 )[0]
>> array([5, 6])
```

> Besides storing whole adjacency matrix information where many entries would be 0 and not important, sometimes edge connection information is stored in a `coordinate format`. In this format the edge-connections are described in tuples and only non-zero entries are populated. This way the representation is sparse and not memory intensive. 

----------------

## Graph neural network implementation

Given the graph, node features, and the node connections with other nodes we can contruct the graph convolution operation to use the geometric information and predict properties of the graph and the nodes. 

**1. Tipf's Graph Convolution Implementation**

Basic implementation of GCN used when making the neural network.

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

Building the total neural network model:

The model consists of 3 GCN parts which you can think of going upto 3 nearest neighbors to account for the local information. Finally the updated nodes features post each convolution is fed in to a fully-connected neural network where the output is one of the 4 classes. 

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
    optimizer.zero_grad() #reset optimizer cache 
    loss=criterion(out[data.train_mask], data.y[data.train_mask]) #estimate loss on ONLY 4 nodes -- mask to identify the nodes 
    loss.backward() #initiate back-prop 
    optimizer.step() #update the NN weights 
    if i % 100==0:
        print("Step: {} Cross Entropy Loss = {}".format(i, loss.item()))
        output_, _ = simple_GCN(node_features)
        visualize_graph(data, output_)
```

**First forward-pass result:**

At first visualizing the output there is not clear distinction in the nodes. The coloring is done as the ground truth labels. 

![first_simple](/img/Simple_GCN/simple/first.png)

**Visualizing post-GNN training:**

Once the weight in the GCN defined above are trained on the node connections and node label and ONLY 4 nodes, the clustering of all nodes in 4 groups becomes apparent. The Class 2 which is the light blue group is the most distinct and it is also the most well separated of the group in the original represtnation too. There is some overlap in the Class 1 3 4 which is captured in the low dimensional as well. However given information of final label of only 4 nodes the GCN does a nice job of clustering all the nodes in their respective 4 clusters. 

![final_simple](/img/Simple_GCN/simple/final_output.png)

----------------

**2. PyTorch Geometric Implementation**

This part is adapted from PyTorch Geometric's tutorial page. [Link](https://colab.research.google.com/drive/1h3-vJGRVloF5zStxL5I0rSy4ZUPNsjy8?usp=sharing#scrollTo=7cjjyFVnpKB0)

In this case we use the `GCN` module built in the `PyTorch Geometric` package. 

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

**Visualizing post-GNN training:**

![pyg_final](/img/Simple_GCN/pyg_gcn/final_output.png)

