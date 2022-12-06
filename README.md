# NeuroHuff-CPI
All the source code and three datasets have been uploaded.

To run the code, please firstly install the python libararies/enviroment by the command 'pip install -r requirements.txt' within conda virtual enviroment, if you would like to manually set up the python version and other dependencies, please use the suggested versions and necessary libraries in the following:

## Setup and dependencies 
Dependencies:
- python >= 3.6
- tensorflow 2.4
- numpy
- sklearn

# Three datasets 
We provides three datasets, namely BindingDB, Celegans and Humans, which roughly cover more than 30,000 balanced samples. The format used in those datasets are like such: 
|  Protein_sequence  |  Compound_SMILES  |  Label  |
| -------------------|-------------------|---------|
|  ABEDSED...        |  C#2OCN...        |  0      |
|  ...               |  ...              |  ...    |
|  CCDJDSE...        |  NCO@2H...        |  1      |


# Run:
We provide an example of how to reproduce the experiments described in the paper. Although it runs for 60 epochs, we find 30 epochs is way enough and all the results in paper are run around 50 epochs. 
You can also directly run `python main.py or python predict.py ` to run the experiments and get the results.


## Contact 
Please feel free to contact me (nick@s.upc.edu.cn), if there is any questions in general or deloying the codes.
