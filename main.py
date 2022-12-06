import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
from model import Net
from pardata import parse_data
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',default='./data')
    #struc params
    parser.add_argument('--window-sizes','-w',type=int,default=15)
    parser.add_argument('--protein-layers','-p',type=int,default=64)
    parser.add_argument('--drug-layers','-c',type=int,default=128)
    parser.add_argument('--fc-layers','-f',type=int,default=64)
    #training params
    parser.add_argument('--lr','-r',default=0.0005,type=float)
    parser.add_argument('--epoch','-e',default=35,type=int)
    #other params
    parser.add_argument('--activation','-a',default='relu',type=str)
    parser.add_argument('--dropout','-D',default=0.2,type=float)
    parser.add_argument('--filters','-F',default=128,type=int)
    parser.add_argument('--batch-size','-b',type=int,default=16)
    parser.add_argument('--decay','-y',default=0.0001,type=float)
    #mode params
    parser.add_argument('--validation',action='store_true')
    args = parser.parse_args()
    ########################################################################
    #Network params
    network_params = {
        'protein_strides': args.window_sizes,
        'protein_layers' : args.protein_layers,
        'drug_layers': args.drug_layers,
        'fc_layers' : args.fc_layers,

        'learning_rate': args.lr,
        'n_epoch': args.epoch,

        'activation' : args.activation,
        'dropout' : args.dropout,
        'filters': args.filters,
        'batch_size': args.batch_size,
        'decay': args.decay,
    }

    print('\t model parameters summary \t')
    print('=====================================')
    for key in network_params.keys():
        print('{:20s}:  {:10s}'.format(key,str(network_params[key])))

    ourmodel = Net(**network_params)
    ourmodel.summary()
    feature = parse_data()

    ourmodel.train(**feature,batch_size=args.batch_size)
