import torch
import pandas as pd


import json

with open('Path', 'r') as file:
    data = json.load(file)
print(len(data))
print(len(list(data.values())[0]))

node_features = torch.tensor(list(data.values()), dtype=torch.float32)

node_features

#pip install torch_geometric

import pandas as pd
import torch

df_train = pd.read_csv("Path")

df_test = pd.read_csv("Path")

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

## test

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

# negative edge for test

filt = df_test["Class Label"] == 0
columns_to_select = ['node1', 'node2']
test_data_neg = df_test[filt][columns_to_select].values

#test_data_int = test_data_neg.astype(int)
test_data_edge_neg = torch.tensor(test_data_neg, dtype=torch.long).t().contiguous()


test_edge_neg_labels = torch.zeros(test_data_edge_neg.size(1))



# model Gat

import torch.nn as nn
from torch_geometric.nn import GATv2Conv,GATConv
import torch.nn.functional as F

class New_GAT_two_layer(torch.nn.Module):

  def __init__(self, dim_in, dim_h, dim_out, heads=1):
    super().__init__()
    self.gat1 = GATv2Conv(dim_in, dim_h, heads=heads)
    #self.layer1 = nn.Linear(dim_h,dim_h)
    #self.gat2 = GATv2Conv(dim_h*heads, dim_out, heads=1)
    self.gat2 = GATv2Conv(dim_out*heads, dim_out, heads=1)
    self.layer1 = nn.Linear(dim_h,dim_out)
    self.layer2 = nn.Linear(dim_out,dim_out)
    #self.batch_norm1 = nn.BatchNorm1d(dim_h)
    #self.batch_norm2 = nn.BatchNorm1d(dim_out)
    #self.batch_norm3 = nn.BatchNorm1d(dim_out)
    #self.update = nn.Linear(dim_h*heads+dim_out,dim_out)

  def encode(self,x,edge_val):
     #h = F.dropout(x, p=0.6,training=self.training)
     h = self.gat1(x, edge_val)
     #r h = self.batch_norm1(h)
     h = F.relu(h)
     val1 = F.relu(self.layer1(h))
     h = F.dropout(val1, p=0.5, training=self.training)
     h = self.gat2(h, edge_val)
     #r h = self.batch_norm2(h)
     #h = self.batch_norm3(h)
     h = F.relu(h)	
     #h=self.batch_norm3(h)			
     val2 = F.relu(self.layer2(h))
     h = F.dropout(val2, p=0.5, training=self.training)
     #val3 = F.dropout(val2, p=0.5, training=self.training)
     #val = F.relu(self.update(h))
     return h
  def decode(self,updated,edge_val):
    val = (updated[edge_val[0]]*updated[edge_val[1]]).sum(dim=-1)

    return val







# model Gat

model_gat = New_GAT_two_layer(trainset.num_features,68,34)
#model_gat.to(device)
optimizer = torch.optim.Adam(model_gat.parameters(),lr=0.01,weight_decay=1e-4)
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
  model_gat.eval()
  optimizer.zero_grad()
  z = model_gat.encode(torch.Tensor(func_normalize(trainset.x)),trainset.edge_index)
  out = model_gat.decode(z, train_edge_label_index).view(-1)
  train_out.append(out)
  loss = F.binary_cross_entropy_with_logits(out,train_edge_label)
  train_loss.append(loss.item())
  predictions = (torch.sigmoid(out) > 0.5).float()
  train_pred.append(predictions)
  #print("tr",train_pred)
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
  model_gat.eval()
  z=model_gat.encode(torch.Tensor(func_normalize(testset.x)),testset.edge_index)
  out = model_gat.decode(z, test_edge_label_index).view(-1)
  loss = F.binary_cross_entropy_with_logits(out,test_edge_label)
  total_loss.append(loss.item())

  predictions = (torch.sigmoid(out) > 0.5).float()
  correct_predictions = (predictions == test_edge_label).float()
  acc = correct_predictions.mean()
  test_accuracies.append(acc)
  test_out.append(predictions)
  #print("ts",test_out)
  conf_matrix = confusion_matrix(test_edge_label, predictions)
  test_confusion_matrix.append(conf_matrix)
  return acc

for epoch in range(1,1000):
  loss=train()
  acc=test()
  print(f"Epoch {epoch}, train_Loss: {loss:.10f}, test_Accuracy: {acc:.4f}")

print(max(train_acc))
#print("pp",train_acc.index(max(train_acc)))
#print("qq", train_out[train_acc.index(max(train_acc))])
#print("ss", train_pred[train_acc.index(max(train_acc))])

print(max(test_accuracies))

import json
import numpy as np

filename = "Path"

#r output_list = train_out[train_acc.index(max(train_acc))].detach().numpy().tolist()
output_list = train_pred[train_acc.index(max(train_acc))].detach().numpy().tolist()



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

filename_test = "Path"

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

torch.save(model_gat.state_dict(), "Path")



