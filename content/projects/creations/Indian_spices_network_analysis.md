---
description: Analyze 6000+ recipes to tease out relations between most commonly used Indian spice. 
fact: 
featured: true
image: /img/food_graph/frequency_plot.png
link: https://github.com/pgg1610/misc_notebooks/blob/master/food_database/food_relations.ipynb
tags:
- Web scarping
- Python
- Network analysis
title: Indian spices network analysis
---

Indian food, like most of the food from the tropical region, uses a variety of different spices. Historically, the spices are thought to have been a way to increase the shelf-life and keep pests away. Anyways, if you are planning to cook more Indian food lately, it would be great to know which spices would give you most bang for your buck. More so, which spices are often used together? What is the relation between each of them? 

Recently I found a dataset from Kaggle which tabulated 6000+ recipes from https://www.archanaskitchen.com/. Using this data as base collection of recipes representing most of the indian food, I analyze which spices occur most freqeuntly and which spices are most connected to each other. 

* Dataset for Indian recipe: This dataset 6000+ recipe scrapped from: [Dataset](https://www.kaggle.com/kanishk307/6000-indian-food-recipes-dataset)

### Key results:
![freq_plot](/img/food_graph/frequency_plot.png){width=20%}

Plot showing the number of times a spices occurs in the list of 6000+ recipes. I am showing only top few entries for clarity.

![graph](/img/food_graph/Graph.png)

Circular graph amongst the indian spices. Size of the node is the relevance of that spice in Indian cuisine i.e. number of times that spice occured in entire recipe collection. Edge color/width shows the strength of the connection amongst different spices. Right away it is seen that: Turmeric, chilli powder, and cumin are always used together. 

![correlation plot](/img/food_graph/heatmap.png)

Correlation plot showing binary correlations between different spices. 