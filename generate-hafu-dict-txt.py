import json
all_data = []
with open('./data/human/data.txt') as f:
    lines = f.readlines()
for line in lines:
    all_data.append(line.strip('\n').split(' '))
#####################################################################################
letter = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
count = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
for i in range(len(all_data)):
        for k in range(len(letter)):
            count[k] += all_data[i][1].count(letter[k])
dit = dict(zip(letter,count))
sort_dit = sorted(dit.items(),key=lambda item:item[1],reverse=True)
print(sort_dit)
##################################################################################
final_dict = {}
for i in range(len(sort_dit)):
    final_dict[sort_dit[i][0]] = i
print('Hafuman frequency letter dict:',final_dict)
###################################################################################
bin_dict = {}
temp = 1
for x in final_dict:
    bin_dict[x]= bin(temp)[2:].rjust(5,'0')
    temp +=1
for id,idxx in enumerate(bin_dict):
    a = bin_dict[idxx]
    l = []
    for i in range(len(a)):
        l.append(int(a[i]))
    bin_dict[idxx] = l
print('Hafuman frequency Binary dict:',bin_dict)
savepath = './amino.json'
with open(savepath, 'w') as f:
    json.dump(bin_dict, f)
print('generation is completed!')