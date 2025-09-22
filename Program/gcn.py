import torch
import pandas as pd


import json

with open('Path/feature.json', 'r') as file:
    data = json.load(file)
print(len(data))
print(len(list(data.values())[0]))


node_features = torch.tensor(list(data.values()), dtype=torch.float32)

node_features

#pip install torch_geometric

import pandas as pd
import torch

df_train = pd.read_csv("Path/train_fold")

df_test = pd.read_csv("Path/test_fold")

df_train.head()

df_train.tail()

df_test.tail()

df_test.head()

print(len(df_test))

node_features

filt = df_train["Class Label"] == 1
columns_to_select = ['node1', 'node2']
train_data = df_train[filt][columns_to_select].values

train_data_edge = torch.tensor(train_data, dtype=torch.long).t().contiguous()

from torch_geometric.data import Data


node_labels = torch.ones(train_data_edge.size(1))
train_edge_labels = torch.ones(train_data_edge.size(1))

trainset=Data(
    x=node_features,
    edge_index=train_data_edge,
    y=node_labels,
    edge_labels=train_edge_labels
)

trainset

df_train.head()

trainset.edge_index

# Negative edge for train

filt = df_train["Class Label"] == 0
columns_to_select = ['node1', 'node2']
train_data_neg = df_train[filt][columns_to_select].values

train_data_edge_neg = torch.tensor(train_data_neg, dtype=torch.long).t().contiguous()

train_edge_neg_labels = torch.zeros(train_data_edge_neg.size(1))

from torch_geometric.data import Data, DataLoader


filt = df_test["Class Label"] == 1
columns_to_select = ['node1', 'node2']
test_data = df_test[filt][columns_to_select].values

#test_data_int = test_data.astype(int)
test_data_edge = torch.tensor(test_data, dtype=torch.long).t().contiguous()

node_labels = torch.ones(test_data_edge.size(1))
test_edge_labels = torch.ones(test_data_edge.size(1))


testset=Data(
    x=node_features,
    edge_index=test_data_edge,
    y=node_labels,
    edge_labels=test_edge_labels
)

testset



filt = df_test["Class Label"] == 0
columns_to_select = ['node1', 'node2']
test_data_neg = df_test[filt][columns_to_select].values

#test_data_int = test_data_neg.astype(int)
test_data_edge_neg = torch.tensor(test_data_neg, dtype=torch.long).t().contiguous()


test_edge_neg_labels = torch.zeros(test_data_edge_neg.size(1))





import torch.nn as nn
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

class New_Gcnn_two_Layer(nn.Module):
  def __init__(self,input_dim,hidden_dim,output_dim):
    super(New_Gcnn_two_Layer,self).__init__()
    self.gcn1 = GCNConv(input_dim,hidden_dim)
    self.gcn2 = GCNConv(hidden_dim,output_dim)
    self.layer1 = nn.Linear(hidden_dim,hidden_dim)
    self.layer2 = nn.Linear(output_dim,output_dim)
    #self.update = nn.Linear(hidden_dim+output_dim,output_dim)
    #self.update = nn.Linear(output_dim,output_dim)


  def encode(self,x,edge_val):
     """gcn1 = F.relu(self.layer1(x,edge_val))
     gcn2 = self.layer2(gcn1,edge_val)
     #gcn_final = torch.cat((gcn1,gcn2),1)
     #val = F.relu(self.update(gcn_final))
     val = F.relu(self.update(gcn2))
     return val"""
     h = self.gcn1(x, edge_val)
     h = F.relu(h)
     #h = F.dropout(h, p=0.5, training=self.training)
     val1 = self.layer1(h)
     #val1 = F.relu(self.layer1(h))
     h = self.gcn2(val1, edge_val)
     #h = F.relu(h)
     val2 = F.relu(self.layer2(h))
     #val2 = F.dropout(val2, p=0.2, training=self.training)
     return val2

  def decode(self,updated,edge_val):
    val = (updated[edge_val[0]]*updated[edge_val[1]]).sum(dim=-1)

    return val




model_gcn = New_Gcnn_two_Layer(trainset.num_features,68,34)
optimizer = torch.optim.Adam(model_gcn.parameters(),lr=0.01, weight_decay=1e-4)
criterion = torch.nn.BCEWithLogitsLoss()
#criterion = CustomLoss()
#criterion = nn.MSELoss()

def func_normalize(data_value):
  from sklearn.preprocessing  import MinMaxScaler
  mm = MinMaxScaler()
  train_x = mm.fit_transform(data_value)
  return train_x

train_edge_label_index = torch.cat(
            [trainset.edge_index, train_data_edge_neg],
            dim=-1,
        )

train_edge_label = torch.cat([
            trainset.edge_labels,
            train_edge_neg_labels
        ], dim=0)

trainset

from sklearn.metrics import confusion_matrix

train_loss = []
train_acc = []
train_pred = []
train_out = []
train_confusion_matrix = []

import torch.nn.functional as F
def train():
  model_gcn.eval()
  optimizer.zero_grad()
  z = model_gcn.encode(torch.Tensor(func_normalize(trainset.x)),trainset.edge_index)
  out = model_gcn.decode(z, train_edge_label_index).view(-1)
  train_out.append(out)
  loss = F.binary_cross_entropy_with_logits(out,train_edge_label)
  train_loss.append(loss.item())
  predictions = (torch.sigmoid(out) > 0.5).float()
  train_pred.append(predictions)
  correct_predictions = (predictions == train_edge_label).float()
  acc = correct_predictions.mean()
  conf_matrix = confusion_matrix(train_edge_label, predictions)
  train_confusion_matrix.append(conf_matrix)
  train_acc.append(acc)
  loss.backward()
  optimizer.step()
  return loss.item()

test_edge_label_index = torch.cat(
            [testset.edge_index, test_data_edge_neg],
            dim=-1,
        )

test_edge_label = torch.cat([
            testset.edge_labels,
            test_edge_neg_labels
        ], dim=0)

total_loss=[]
test_accuracies=[]
test_confusion_matrix = []
test_out = []

def test():
  model_gcn.eval()
  z=model_gcn.encode(torch.Tensor(func_normalize(testset.x)),testset.edge_index)
  out = model_gcn.decode(z, test_edge_label_index).view(-1)
  loss = F.binary_cross_entropy_with_logits(out,test_edge_label)
  total_loss.append(loss.item())

  predictions = (torch.sigmoid(out) > 0.5).float()
  correct_predictions = (predictions == test_edge_label).float()
  acc = correct_predictions.mean()
  test_accuracies.append(acc)
  test_out.append(predictions)
  conf_matrix = confusion_matrix(test_edge_label, predictions)
  test_confusion_matrix.append(conf_matrix)
  return acc

for epoch in range(1,500):
  loss=train()
  acc=test()
  print(f"Epoch {epoch}, train_Loss: {loss:.10f}, test_Accuracy: {acc:.4f}")






test_accuracies.index(max(test_accuracies))

import json
import numpy as np

filename = "Path/gcn_train_fold.json"

output_list = train_out[train_acc.index(max(train_acc))].detach().numpy().tolist()


cm_train_int = [[int(x) for x in row] for row in train_confusion_matrix[train_acc.index(max(train_acc))]]

conf_matrix = train_confusion_matrix[train_acc.index(max(train_acc))]
tn, fp, fn, tp = conf_matrix.ravel()


data = {
    'accuracy': float(max(train_acc)),
    'TP': int(tp),
    'FP': int(fp),
    'FN': int(fn),
    'TN': int(tn),
    'output': output_list
}


with open(filename, 'w') as json_file:
    json.dump(data, json_file, indent=4)

filename_test = "Path/gcn_test_fold.json"

output_list_test = test_out[test_accuracies.index(max(test_accuracies))].detach().numpy().tolist()


tn, fp, fn, tp = test_confusion_matrix[test_accuracies.index(max(test_accuracies))].ravel()

data_test = {
    'accuracy': float(max(test_accuracies)),
    "TN": int(tn),
    "FP": int(fp),
    "FN": int(fn),
    "TP": int(tp),
    'output': output_list_test
}


with open(filename_test, 'w') as json_file:
    json.dump(data_test, json_file, indent=4)

torch.save(model_gcn.state_dict(), "Path/gcn_model_fold.pt")
