



import torch
import pandas as pd

pip install torch_geometric



df_gat_train = pd.read_json("Path/gat_train_fold_number.json")

df_gcn_train = pd.read_json("Path/gcn_train_fold_number.json")



df_gat_test = pd.read_json("Path/gat_test_fold_number.json")

df_gcn_test = pd.read_json("Path/gcn_test_fold_number.json")




out_train_gat = torch.tensor(df_gat_train["output"].values)
out_train_gcn = torch.tensor(df_gcn_train["output"].values)




out_gat_test = torch.tensor(df_gat_test["output"].values)
out_gcn_test = torch.tensor(df_gcn_test["output"].values)


out_train_gat_ppi

out_train_gat



train_pos = torch.ones(int(len(out_train_gat) / 2))
train_neg = torch.zeros(int(len(out_train_gat) / 2))
train_labels = torch.cat((train_pos, train_neg), dim=0)


test_pos = torch.ones(int(len(out_gat_test) / 2))
test_neg = torch.zeros(int(len(out_gat_test) / 2))
test_labels = torch.cat((test_pos, test_neg), dim=0)

print(len(test_labels))



from sklearn.neural_network import MLPClassifier
import numpy as np

stck_x = np.column_stack((out_train_gat.float(),out_train_gcn.float()))



train_edge_label_np = train_labels.numpy()



test_edge_label_np = test_labels.numpy()



train_edge_label_np

len(test_edge_label_np)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(stck_x,train_edge_label_np)



stck_test_x = np.column_stack((out_gat_test,out_gcn_test))

final_pred_lr = lr.predict(stck_test_x)



stck_test_x
stck_test_x = stck_test_x.astype(np.float32)
stck_test_x


from sklearn.metrics import accuracy_score

print(accuracy_score(test_edge_label,final_pred_lr))



import torch.nn as nn
class MLP(nn.Module):
  def __init__(self,dim_in,dim_h,dim_out):
    super(MLP,self).__init__()
    self.layer1 = nn.Linear(dim_in,dim_h)
    self.layer2 = nn.Linear(dim_h,dim_out)
    self.layer3 = nn.Linear(dim_out,dim_out)
  def forward(self,out):
    h = F.dropout(out, p=0.1, training=self.training)
    #h = F.dropout(out, p=0.2, training=self.training)
    h = self.layer1(h)
    h = F.relu(h)
    h = F.dropout(h, p=0.1, training=self.training)
    #h = F.dropout(h, p=0.2, training=self.training)
    h = F.relu(self.layer2(h))
    #h = F.relu(h)
    h = F.dropout(h, p=0.2, training=self.training)
    #h = F.dropout(h, p=0.5, training=self.training)
    #h = self.layer2(h)
    #val = (h[edge_index[0]]*h[edge_index[1]]).sum(dim=-1)
    return h





mlp = MLP(2,1,1)
optimizer = torch.optim.Adam(mlp.parameters(),lr = 0.01, weight_decay=2e-4)
criterion = torch.nn.BCEWithLogitsLoss()
mlp.eval()

testset




train_ensemble_acc = []
train_ensemble_pred = []
train_ensemble_out = []

def train():
  optimizer.zero_grad()
  output = mlp(torch.tensor(func_normalize(stck_x)))
  output = output.view(len(stck_x))
  train_ensemble_out.append(output)
  loss = F.binary_cross_entropy_with_logits(output,train_labels)
  loss.backward()
  optimizer.step()
  predictions = (torch.sigmoid(output) > 0.5).float()
  train_ensemble_pred.append(predictions)
  correct_predictions = (predictions == train_labels).float()
  acc = correct_predictions.mean()
  train_ensemble_acc.append(acc)
  return loss.item()





test_ensemble_acc = []
test_ensemble_pred = []
test_ensemble_out = []

def test():
  output_pred = mlp(torch.tensor(func_normalize(stck_test_x)))
  output_pred = output_pred.view(len(stck_test_x))
  loss = F.binary_cross_entropy_with_logits(output_pred,test_labels)
  predictions = (torch.sigmoid(output_pred) > 0.5).float()
  test_ensemble_pred.append(predictions)
  correct_predictions = (predictions == test_labels).float()
  acc = correct_predictions.mean()
  test_ensemble_acc.append(acc)
  test_ensemble_out.append(predictions)
  return acc




for epoch in range(1,400):
  loss=train()
  acc=test()
  print(f"Epoch {epoch}, train_Loss: {loss:.10f}, test_Accuracy: {acc:.4f}")

train_ensemble_acc[-1]




from sklearn.metrics import confusion_matrix
cm_train = confusion_matrix(train_labels,train_ensemble_pred[train_ensemble_acc.index(max(train_ensemble_acc))])
print(cm_train)


torch.save(mlp.state_dict(), "Path/ensemble_model_fold_number.pt")

len(list(train_ensemble_out[-1].detach().numpy()))

train_ensemble_out[-1]



tn, fp, fn, tp = cm_train.ravel()

cm_train



import json
import numpy as np

filename = "Path/ensemble_train_fold_number.json"

output_list = train_ensemble_out[train_ensemble_acc.index(max(train_ensemble_acc))].detach().numpy().tolist()



data = {
    'accuracy': float(max(train_ensemble_acc)),
    'TP': int(tp),
    'FP': int(fp),
    'FN': int(fn),
    'TN': int(tn),
    'output': output_list
}



with open(filename, 'w') as json_file:
    json.dump(data, json_file, indent=4)

#run

cm_test = confusion_matrix(test_labels_ppi,test_ensemble_pred[test_ensemble_acc.index(max(test_ensemble_acc))])

#run

tn, fp, fn, tp = cm_test.ravel()

#run

import json
import numpy as np

filename = "Path/ensemble_test_fold_number.json"

output_list_test = test_ensemble_out[test_ensemble_acc.index(max(test_ensemble_acc))].detach().numpy().tolist()



data = {
    'accuracy': float(max(test_ensemble_acc)),
    'TP': int(tp),
    'FP': int(fp),
    'FN': int(fn),
    'TN': int(tn),
    'output': output_list_test
}



with open(filename, 'w') as json_file:
    json.dump(data, json_file, indent=4)



df_ensemble = pd.read_json("Path")

df_ensemble
