#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import pandas as pd


# In[ ]:


import pandas as pd

df_hold_pos = pd.read_csv("positive_hold_out.csv")

df_hold_neg = pd.read_csv("negative_hold_out.csv")


# In[ ]:


df_hold_pos["Class Label"] = 1
df_hold_neg["Class Label"] = 0


# In[ ]:


pip install torch_geometric


# In[ ]:


df_node_features = pd.read_csv("/content/drive/MyDrive/orbit-counts (1).txt",names= ["col"])


# In[ ]:


import pandas as pd
import torch
import json


# In[ ]:


lst_node_feature = []
for string_data in df_node_features["col"].values:
  integer_data = [int(x) for x in string_data.split()]
  lst_node_feature.append(integer_data)


# In[ ]:


node_features = torch.tensor(lst_node_feature, dtype=torch.float32)


# In[ ]:


filt = df_hold_out["Class Label"] == 1
columns_to_select = ['node1', 'node2']
hold_out_data = df_hold_out[filt][columns_to_select].values


# In[ ]:


hold_out_data_edge = torch.tensor(hold_out_data, dtype=torch.long).t().contiguous()


# In[ ]:


from torch_geometric.data import Data


node_labels = torch.ones(hold_out_data_edge.size(1))
hold_out_edge_labels = torch.ones(hold_out_data_edge.size(1))

hold_outset=Data(
    x=node_features,
    edge_index=hold_out_data_edge,
    y=node_labels,
    edge_labels=hold_out_edge_labels
)


# In[ ]:


hold_outset


# In[ ]:


# Negative edge for hold_out

filt = df_hold_out["Class Label"] == 0
columns_to_select = ['node1', 'node2']
hold_out_data_neg = df_hold_out[filt][columns_to_select].values


# In[ ]:


hold_out_data_edge_neg = torch.tensor(hold_out_data_neg, dtype=torch.long).t().contiguous()


# In[ ]:


hold_out_edge_neg_labels = torch.zeros(hold_out_data_edge_neg.size(1))


# In[ ]:


from torch_geometric.data import Data, DataLoader


# In[ ]:


def func_normalize(data_value):
  from sklearn.preprocessing  import MinMaxScaler
  mm = MinMaxScaler()
  train_x = mm.fit_transform(data_value)
  return train_x


# In[ ]:


hold_out_edge_label_index = torch.cat(
            [hold_outset.edge_index, hold_out_data_edge_neg],
            dim=-1,
        )

hold_out_edge_label = torch.cat([
            hold_outset.edge_labels,
            hold_out_edge_neg_labels
        ], dim=0)


# In[ ]:


hold_outset


# In[ ]:





# In[ ]:


# model GCN

import torch.nn as nn
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

class New_Gcnn_two_Layers(nn.Module):
  def __init__(self,input_dim,hidden_dim,output_dim):
    super(New_Gcnn_two_Layers,self).__init__()
    self.gcn1 = GCNConv(input_dim,hidden_dim)
    self.gcn2 = GCNConv(hidden_dim,output_dim)
    self.layer1 = nn.Linear(hidden_dim,hidden_dim)
    self.layer2 = nn.Linear(output_dim,output_dim)
    #self.update = nn.Linear(hidden_dim+output_dim,output_dim)
    #self.update = nn.Linear(output_dim,output_dim)


  def encode(self,x,edge_val):
     h = self.gcn1(x, edge_val)
     h = F.relu(h)
     #h = F.dropout(h, p=0.5, training=self.training)
     val1 = F.relu(self.layer1(h))
     h = self.gcn2(val1, edge_val)
     h = F.relu(h)
     val2 = F.relu(self.layer2(h))
     #val2 = F.dropout(val2, p=0.2, training=self.training)
     return val2

 
  def decode(self,updated,edge_val):
    val = (updated[edge_val[0]]*updated[edge_val[1]]).sum(dim=-1)

    return val


# In[ ]:





# In[ ]:


# model Gat

import torch.nn as nn
from torch_geometric.nn import GATv2Conv,GATConv
import torch.nn.functional as F

class New_GAT_two_layers(torch.nn.Module):

  def __init__(self, dim_in, dim_h, dim_out, heads=1):
    super().__init__()
    self.gat1 = GATv2Conv(dim_in, dim_h, heads=heads)
    self.gat2 = GATv2Conv(dim_out*heads, dim_out, heads=1)
    self.layer1 = nn.Linear(dim_h,dim_out)
    self.layer2 = nn.Linear(dim_out,dim_out)
    #self.update = nn.Linear(dim_h*heads+dim_out,dim_out)

  def encode(self,x,edge_val):
     #h = F.dropout(x, p=0.2, training=self.training)
     h = self.gat1(x, edge_val)
     h = F.relu(h)
     val1 = F.relu(self.layer1(h))
     h = F.dropout(val1, p=0.5, training=self.training)
     h = self.gat2(h, edge_val)
     h = F.relu(h)
     val2 = F.relu(self.layer2(h))
     h = F.dropout(val2, p=0.5, training=self.training)
     return h
  def decode(self,updated,edge_val):
    val = (updated[edge_val[0]]*updated[edge_val[1]]).sum(dim=-1)

    return val


# In[ ]:





# In[ ]:


import torch

model_gat = New_GAT_two_layers(hold_outset.num_features,68,34)
model_gat.load_state_dict(torch.load('Model/model_gat_fold_number .pt'))
model_gat.eval()




with torch.no_grad():
  z=model_gat.encode(torch.Tensor(func_normalize(hold_outset.x)),hold_outset.edge_index)
  out_gat_hold_out = model_gat.decode(z, hold_out_edge_label_index).view(-1)


# In[ ]:


prediction_hold_out_gat = (torch.sigmoid(out_gat_hold_out) > 0.5).float()
correct_prediction_hold_out_gat = (prediction_hold_out_gat == hold_out_edge_label).float()
acc_gat_hold_out = correct_prediction_hold_out_gat.mean()
print(acc_gat_hold_out)


# In[ ]:


import torch

model_gcn = New_Gcnn_two_Layers(hold_outset.num_features,68,34)
model_gcn.load_state_dict(torch.load('Model/gcn_train_fold_number.pt'))
model_gcn.eval()





with torch.no_grad():
  z=model_gcn.encode(torch.Tensor(func_normalize(hold_outset.x)),hold_outset.edge_index)
  out_gcn_hold_out = model_gcn.decode(z, hold_out_edge_label_index).view(-1)


# In[ ]:


prediction_hold_out_gcn = (torch.sigmoid(out_gcn_hold_out) > 0.5).float()
correct_prediction_hold_out_gcn = (prediction_hold_out_gcn == hold_out_edge_label).float()
acc_gcn_hold_out = correct_prediction_hold_out_gcn.mean()
print(acc_gcn_hold_out)


# In[ ]:


from sklearn.metrics import confusion_matrix
cm_hold_out_gcn = confusion_matrix(hold_out_edge_label ,prediction_hold_out_gcn)
print(cm_hold_out_gcn)


# In[ ]:


from sklearn.metrics import confusion_matrix
cm_hold_out_gat = confusion_matrix(hold_out_edge_label,prediction_hold_out_gat)
print(cm_hold_out_gat)


# In[ ]:


import json
import numpy as np

filename = "gat_hold_out_fold_number.json"

output_list = out_gat_hold_out.detach().numpy().tolist()


cm_hold_out_int = [[int(x) for x in row] for row in cm_hold_out_gat]


data = {
    'accuracy': float(acc_gat_hold_out),
    'TP': cm_hold_out_int[0][0],
    'FP': cm_hold_out_int[0][1],
    'FN': cm_hold_out_int[1][0],
    'TN': cm_hold_out_int[1][1],
    'output': output_list
}


with open(filename, 'w') as json_file:
    json.dump(data, json_file, indent=4)


# In[ ]:


import json
import numpy as np

filename = "gcn_hold_out_fold_number.json"

output_list = out_gcn_hold_out.detach().numpy().tolist()


cm_hold_out_int = [[int(x) for x in row] for row in cm_hold_out_gcn]


data = {
    'accuracy': float(acc_gcn_hold_out),
    'TP': cm_hold_out_int[0][0],
    'FP': cm_hold_out_int[0][1],
    'FN': cm_hold_out_int[1][0],
    'TN': cm_hold_out_int[1][1],
    'output': output_list
}


with open(filename, 'w') as json_file:
    json.dump(data, json_file, indent=4)


# In[ ]:




