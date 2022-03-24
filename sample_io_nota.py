import json
import sys
import random

if(len(sys.argv) != 8):
    print("Usage: python sample_io.py filename.json size N K Rate seed input/output")
filename = sys.argv[1]
size = int(sys.argv[2])
N = int(sys.argv[3])
K = int(sys.argv[4])
Rate = float(sys.argv[5])
seed = int(sys.argv[6])
io = sys.argv[7]
random.seed(seed)

whole_division = json.load(open(filename))
relations = whole_division.keys()


input_data = []
output_data = []
Nota_size = int(size*Rate)
for i in range(Nota_size):
    sampled_relation = random.sample(relations, N)
    output_data.append(-1)
    target_relation = random.sample(relations, 1)
    while target_relation[0] in sampled_relation:
        target_relation = random.sample(relations, 1)
    meta_train = [random.sample(whole_division[i], K)
                  for i in sampled_relation]
    meta_test = random.choice(whole_division[target_relation[0]])
    input_data.append({"meta_train": meta_train,
                      "meta_test": meta_test, "relation": sampled_relation})

for i in range(size-Nota_size):
    sampled_relation = random.sample(relations, N)
    target = random.choice(range(len(sampled_relation)))
    output_data.append(target)
    target_relation = sampled_relation[target]
    meta_train = [random.sample(whole_division[i], K)
                  for i in sampled_relation]
    meta_test = random.choice(whole_division[target_relation])
    input_data.append({"meta_train": meta_train,
                      "meta_test": meta_test, "relation": sampled_relation})

for i in range(1, size):
    j = random.randint(0, i-1)
    input_data[i], input_data[j] = input_data[j], input_data[i]
    output_data[i], output_data[j] = output_data[j], output_data[i]

if(io == "input"):
    json.dump(input_data, sys.stdout)
    # print(input_data)
else:
    json.dump(output_data, sys.stdout)
    # print(output_data)
