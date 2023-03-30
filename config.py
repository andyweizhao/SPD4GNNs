
import argparse

def load_arguments():
	parser = argparse.ArgumentParser()
	parser.add_argument("--hidden_dims", default=3, type=int, help="the dimension of SPD.")
	parser.add_argument("--dataset", default='disease_nc', type=str) 
	parser.add_argument("--model", default='spdgcn', type=str)
	parser.add_argument("--batchsize", default=-1, type=int) 
	parser.add_argument("--patience", default=200, type=int) 
	parser.add_argument("--classifier", default='linear', type=str)
	parser.add_argument("--manifold", default='spd', type=str, choices=["euclidean", "spd"])    
	parser.add_argument("--learningrate", default=0.01, type=float) 
	parser.add_argument("--dropout", default=0, type=float)
	parser.add_argument("--weight_decay", default=5e-4, type=float)

	parser.add_argument("--spd_norm", default=None, type=str)
	parser.add_argument("--vec2sym", default='squared', type=str)

	parser.add_argument("--transform", default="qr", type=str, help="the choice of producing isometry map",
	                        choices=["cayley", "qr"]) 

	parser.add_argument("--epoch", default=300, type=int)

	parser.add_argument("--use-feats", default=1, type=float, help='whether to use node features or not')

	parser.add_argument("--normalize-feats", default=0, type=float, help='whether to normalize input node features')

	parser.add_argument("--normalize-adj", default=1, type=float, help='whether to row-normalize the adjacency matrix')
	parser.add_argument("--split-seed", default=1234, type=float, help='seed for data splits (train/test/val)')

	parser.add_argument("--has_norm", default=1, type=int)
	parser.add_argument("--has_bias", default=1, type=int)

	parser.add_argument("--val_every", default=1, type=int)

	parser.add_argument("--c", default=0.0005, type=float)

	return parser.parse_args()
