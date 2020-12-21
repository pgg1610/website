---
date: "2020-11-08"
description: Analyze 6000+ recipes to find relations between spices most frequently used in Indian cuisine. 
pubtype: Jupyter Notebook
fact: 
featured: true
image: /img/food_graph/frequency_plot.png
link: https://github.com/pgg1610/misc_notebooks/blob/master/food_database/food_relations.ipynb
tags:
- Web scraping
- Python
- Network analysis
title: Analysis on Spice Use in Indian Food
---

Indian food, like most of the food from the tropical region, uses a variety of different spices. Historically, the spices are thought to have been a way to increase the shelf-life and keep pests away. Whatever be the reason they sure are indespensible to food -- no matter which cuisine you're making. Now, if you are planning to cook more Indian food lately, it would be great to know which spices would give you most bang for your buck. More so, which spices are often used together? What is the relation between each of them? 

Recently I found a dataset from Kaggle which tabulated 6000+ recipes from https://www.archanaskitchen.com/. Using this data as base collection of recipes representing most of the indian food, I analyze which spices occur most freqeuntly and which spices are most connected to each other. 

* Dataset for Indian recipe: This dataset 6000+ recipe scrapped from: [Dataset](https://www.kaggle.com/kanishk307/6000-indian-food-recipes-dataset)

The dataset from the Kaggle csv has entries as follows: 
![df_first](/img/food_graph/dataset_first_img.png)

Some entries in the `TranslatedIngredients` have non-english entries. To filter those out I made the following function: 
```python
def filter_english(string):
    try:
        string.encode('utf-8').decode('ascii')
        out = True
    except UnicodeDecodeError: 
        out = False
    return out
```

Next for consistent tabulation I needed a list of spices to look for. Wikipedia has a page on Indian spices which lists various spices used in Indian cuisin [(Link)](https://en.wikipedia.org/wiki/List_of_Indian_spices). I use this list to search names of spices in the recipe entries. 

One more step is editing the spices so that my string counter can find different versions of the same spice.

```python 
spices_list = spices_list.str.replace('amchoor', 'amchur/amchoor/mango extract') \
                    .replace('asafoetida', 'asafetida/asafoetida/hing') \
                    .replace('thymol/carom seed', 'ajwain/thymol/carom seed') \
                    .replace('alkanet root', 'alkanet/alkanet root') \
                    .replace('chilli powder', 'red chilli powder/chilli powder/kashmiri red chilli powder') \
                    .replace('celery / radhuni seed', 'celery/radhuni seed') \
                    .replace('bay leaf, indian bay leaf', 'bay leaf/bay leaves/tej patta') \
                    .replace('curry tree or sweet neem leaf', 'curry leaf/curry leaves') \
                    .replace('fenugreek leaf', 'fenugreek/kasoori methi') \
                    .replace('nigella seed', 'nigella/black cumin') \
                    .replace('ginger', 'dried ginger/ginger powder') \
                    .replace('cloves', 'cloves/laung') \
                    .replace('green cardamom', 'cardamom/green cardamom/black cardamom')\
                    .replace('indian gooseberry', 'indian gooseberry/amla')\
                    .replace('coriander seed', 'coriander seed/coriander powder')\
                    .replace('cumin seed', 'cumin powder/cumin seeds/cumin/jeera')
```

For every food entries I consider the `TranslatedIngredient` column: 
![df_translate](/img/food_graph/translated_entries.png)

I used regular expression to search for spice names in the entries
```python 
import re 

def search_spice(ingredient_string, spice_string):
    spice_list = spice_string.split('/')
    for _spice in spice_list:
        if re.search(_spice.lower(), ingredient_string.lower()):
            return True
            break
```

```python 
food_spice_mix[spices_list.to_list()] = 42 
for row, values in food_spice_mix.iterrows():
    for spice_entry in spices_list:
        if search_spice(values['TranslatedIngredients'], spice_entry):
            food_spice_mix.loc[row, spice_entry] = 1
        else:
            food_spice_mix.loc[row, spice_entry] = 0
```

This way each food item is searched for a particular spice. A one-hot type binary list is made for every food item. 

Based on this binary entries we can create a adjacency matrix and a frequency plot counting the occurence of every spice in entirety of the food recipe corpus. 

```python 
spice_col_name = spices_list
spice_adj_freq = pd.DataFrame(np.zeros(shape=(len(spices_list),len(spices_list))), columns= spice_col_name, index=spice_col_name)

for row, value in food_spice_mix.iterrows():
    for i in spice_col_name:
        for j in spice_col_name:
            if (value[i] == 1) & (value[j] == 1):
                spice_adj_freq.loc[i,j] += 1

spice_adj_freq = spice_adj_freq/len(food_spice_mix) * 100
```

Using frequency adjacency matrix we can plot a heatmap showing the pair-wise occurence for a given pair of spices. The idea with such an analysis is that if we can check the variation of Spice 1 with all the other spices in the list and compare that to Spice 2's variation with all the other spices in the list, if spice 1 and spice 2 should have similar variation. 

![heat_map2](/img/food_graph/heatmap.png)

This map itself is quite interesting. The color intensity of each title shows the frequency that pair of spice occurred together in a recipe. Brighter the color higher their occurence together. 
Some prominent spice pairs which show similarity are:
1. Curry leaves and Mustard seeds 
3. Tumeric and Chilli Powder 

Some pair of spices never occur together: 
1. Saffron and Fenugreek seeds 
2. Nutmeg and Mustard Seeds 

Those who cook or know indian recipes would see that these pairs make sense and thereby validate the correlation seen from corpus of Indian recipes. 

With that analysis, we can go a step further and analyze this information in form of a circular network graph. 

```python 
nodes_data = [(i, {'count':spice_adj_freq.loc[i, i]}) for i in spice_col_name]
edges_data = [] 
for i in temp_name:
    for j in temp_name:
        if i != j:
            if spice_adj_freq.loc[i,j] != 0.0:
                edges_data.append((i, j, {'weight':spice_adj_freq.loc[i,j], 'distance':1}))

#BUILD THE INITIAL FULL GRAPH
G=nx.Graph()
G.add_nodes_from(nodes_data)
G.add_edges_from(edges_data)

#Assigning weights to each node as per occurence 
weights = [G[u][v]['weight'] for u,v in edges]
w_arr = np.array(weights)
norm_weight =  (w_arr - w_arr.min())/(w_arr.max() - w_arr.min())
```

Finally a `networkx` circular graph is made where each node is a spice entry. Each edge between a pair of spice is a connection provided those two spices are found together in a recipe. The size of the node is the frequency of that spice to occur in all of 6000 food recipes. The thickness of the edge connecting a give spice-pair is the normalized frequency that pair occured among 6000 recipes. Representing the analysis this way we find few key takeaways:
1. Tumeric, Mustard Seeds, Chilli Powder, Corriander Seeds, Cumin Seeds, Curry Leaves, Green Chillies, Asafoetida are the key spices in the Indian cuisine. 
2. Most recipes have strong tendancy to use Tumeric + Chilli Powder + Cumin Seed in them. 

![graph](/img/food_graph/food_graphs.png)
