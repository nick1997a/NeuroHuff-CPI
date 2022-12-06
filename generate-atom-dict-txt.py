import json
atm_list = []
from rdkit import Chem
with open('./data/human/data.txt','r') as f:
    lines = f.readlines()
    for i in range(len(lines)):
        temp = lines[i].strip('\n').split(' ')
        result = Chem.MolFromSmiles(temp[0])
        atoms = result.GetAtoms()
        for item in atoms:
            atm_list.append(item.GetSymbol())
count = enumerate(set(atm_list))
atom_dict = {}
for id,idx in count:
    atom_dict[idx]=id+1
with open('./atom.json', 'w') as f:
    json.dump(atom_dict, f)
print('atoms counting is completed!')