import json
import random
from rdkit import Chem
import numpy as np
with open('./atom.json','r') as f1:
    drug_dict = json.load(f1)
def encode_drug(seq,drug_dict):
    temp = [drug_dict[a] for a in seq]
    for i in range(len(temp),100):
        temp.append(0)
    if len(temp)>100:
        temp = temp[:100]
    return np.array(temp)
with open('./amino.json','r') as f2:
    protein_dict = json.load(f2)
def encode_protein(seq,protein_dict):
    kk = []
    for i in range(len(seq)):
        kk.append(np.array(protein_dict[seq[i]]))
    if(len(kk))<1500:
        for i in range(len(seq,),1500):
            kk.append(np.array(protein_dict['0']))
    else:
        kk = kk[:1500]
    return np.array(kk)

def shuffle_data(feature):
    random.shuffle(feature)
    return feature

def parse_data():
    random.seed(9)            ###### seed 9 only works for celegans datasets.
    # please comment out this line of code when reproducing the results of the paper in BindingDB and Humans datasets
    ###############drug embedding feature
    path = 'celegans'
    all = []
    all_drug = []
    with open('./data/'+path+'/data.txt','r') as f:
        lines = f.readlines()
    shuffle_data(lines)
    count = 1
    for line in lines:
        all.append(line.strip('\n').split(' '))
    for data in all:
        print(count)
        count+=1
        mol = Chem.MolFromSmiles(data[0])
        atoms = mol.GetAtoms()
        aaa = [item.GetSymbol() for item in atoms]
        b = encode_drug(aaa,drug_dict)
        all_drug.append(b)
    all_drug = np.array(all_drug)
    ##############drug matrix
    matrix = []
    for line in lines:
        a = line.strip('\n').split(' ')
        smiles = a[0]
        result = Chem.MolFromSmiles(smiles)
        ajx = Chem.GetAdjacencyMatrix(result)
        distance = Chem.GetDistanceMatrix(result)
        fusion = np.matmul(0.5 * ajx, distance)
        if ajx.shape[0] <= 50:
            final = np.pad(array=fusion, pad_width=((0, 50 - distance.shape[0]), (0, 50 - distance.shape[0])))
        else:
            final = fusion[:50, :50]
        matrix.append(final)
    matrix = np.array(matrix)


    #################protein hafuman feature
    all_protein = []
    for data in all:
        cod = encode_protein(data[1],protein_dict)
        all_protein.append(cod)
    all_protein = np.array(all_protein)


    #################label feature
    all_label = []
    for data in all:
        all_label.append(float(data[2]))
    all_label = np.array(all_label)



    return {
        "drug_feature" : all_drug,
        "matrix": matrix,
        "protein_feature": all_protein,
        "Label": all_label,
    }