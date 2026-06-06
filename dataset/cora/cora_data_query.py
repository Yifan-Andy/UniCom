import torch
import random

data_list = torch.load("../cora_pyg.pt")

labels = data_list[2]
torch.manual_seed(0)
random.seed(0)

print(labels, torch.min(labels), torch.max(labels))
num_class = torch.max(labels) - torch.min(labels) + 1
print(num_class)
communities = [[i for i in range(labels.shape[0]) if labels[i]==j] for j in range(num_class)]

print(f"communities {len(communities)}")

######################### Generate transductive training query and ground-truth #########################
selected_queries_train = []
ground_truth_train = []

queries_per_class = 20  # Number of query samples per class

used_query_nodes_train = set()

for class_id in range(num_class):
    class_nodes = communities[class_id]
    for _ in range(queries_per_class):
        num_query_nodes = torch.randint(1, 4, (1,)).item()
        if len(class_nodes) < num_query_nodes:
            selected_nodes = class_nodes  # Use all nodes if not enough
        else:
            selected_nodes = random.sample(class_nodes, k=num_query_nodes)
        selected_queries_train.append(selected_nodes)
        ground_truth_train.append(class_nodes)  # Full community as ground-truth
        used_query_nodes_train.update(selected_nodes)

# Write training queries
query_file_train = open("cora_train.query", "w")
gt_file_train = open("cora_train.gt", "w")

for i in range(len(selected_queries_train)):
    for j in range(len(selected_queries_train[i])):
        query_file_train.write(str(selected_queries_train[i][j]))
        query_file_train.write(" ")
    query_file_train.write("\n")
    for j in range(len(ground_truth_train[i])):
        gt_file_train.write(str(ground_truth_train[i][j]))
        gt_file_train.write(" ")
    gt_file_train.write("\n")

print(f"Generated {len(selected_queries_train)} queries ({queries_per_class} per class).")
print(f"Total {len(selected_queries_train)} train samples")

######################### Generate transductive validation query and ground-truth #########################
selected_queries_val = []
ground_truth_val = []

for i in range(100):
    selected_class = torch.randint(0, num_class, (1,)).item()
    class_nodes = [n for n in communities[selected_class] if n not in used_query_nodes_train]
    if len(class_nodes) == 0:
        continue
    num_node = min(torch.randint(1, 4, (1,)).item(), len(class_nodes))
    selected_nodes = random.sample(class_nodes, num_node)
    selected_queries_val.append(selected_nodes)
    ground_truth_val.append(communities[selected_class])

# Now update used_query_nodes with val after sampling
used_query_nodes_val = set(n for q in selected_queries_val for n in q)
used_query_nodes_total = used_query_nodes_train.union(used_query_nodes_val)

query_file_val = open("cora_val.query", "w")
gt_file_val = open("cora_val.gt", "w")

for i in range(len(selected_queries_val)):
    for j in range(len(selected_queries_val[i])):
        query_file_val.write(str(selected_queries_val[i][j]))
        query_file_val.write(" ")
    query_file_val.write("\n")
    for j in range(len(ground_truth_val[i])):
        gt_file_val.write(str(ground_truth_val[i][j]))
        gt_file_val.write(" ")
    gt_file_val.write("\n")

print(f"Generated {len(selected_queries_val)} validation queries")

######################### Generate transductive testing query and ground-truth #########################
selected_queries_test = []
ground_truth_test = []

for i in range(100):
    selected_class = torch.randint(0, num_class, (1,)).item()
    class_nodes = [n for n in communities[selected_class] if n not in used_query_nodes_total]
    if len(class_nodes) == 0:
        continue
    num_node = min(torch.randint(1, 4, (1,)).item(), len(class_nodes))
    selected_nodes = random.sample(class_nodes, num_node)
    selected_queries_test.append(selected_nodes)
    ground_truth_test.append(communities[selected_class])

query_file = open("cora_test.query", "w")
gt_file = open("cora_test.gt", "w")

for i in range(len(selected_queries_test)):
    for j in range(len(selected_queries_test[i])):
        query_file.write(str(selected_queries_test[i][j]))
        query_file.write(" ")
    query_file.write("\n")
    for j in range(len(ground_truth_test[i])):
        gt_file.write(str(ground_truth_test[i][j]))
        gt_file.write(" ")
    gt_file.write("\n")

print(f"Generated {len(selected_queries_test)} test queries")

######################### Save the full graph edge list #########################
adj = data_list[0]
print(f"adj shape {adj.shape}")
coalesced_tensor = adj.coalesce()
index = coalesced_tensor.indices()

edge_file = open("cora.edges", "w")
for i in range(index.shape[1]):
    if index[0][i].item() != index[1][i].item():
        edge_file.write(str(index[0][i].item()))
        edge_file.write(" ")
        edge_file.write(str(index[1][i].item()))
        edge_file.write("\n")
