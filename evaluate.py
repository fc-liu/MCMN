import json
import sys

if(len(sys.argv) != 3):
    print("Usage: python evaluate.py your_result standard_output")

ur_result = json.load(open(sys.argv[1]))
std_out = json.load(open(sys.argv[2]))

length = len(std_out)
correct = 0
for i in range(length):
    if(ur_result[i] == std_out[i]):
        correct += 1

print(float(correct)/float(length))