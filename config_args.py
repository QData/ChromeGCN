import os.path as path 
import os

def get_args(parser):
	parser.add_argument('-dataroot', type=str, default='/p/qdata/jjl5sw/ChromeGCN/processed_data/')
	# parser.add_argument('-results_dir', type=str, default='/bigtemp/jjl5sw/deepENCODE/results/encode/')
	parser.add_argument('-results_dir', type=str, default='/p/qdata/jjl5sw/ChromeGCN/results/')
	parser.add_argument('-cell_type', type=str, default='GM12878')
	parser.add_argument('-window_size', type=str, default='1000')
	parser.add_argument('-epochs', type=int, default=100)
	parser.add_argument('-batch_size', type=int, default=64)
	parser.add_argument('-test_batch_size', type=int, default=-1)
	parser.add_argument('-d_model', type=int, default=128)
	parser.add_argument('-optim', type=str, choices=['adam', 'sgd'], default='adam')
	parser.add_argument('-optim2', type=str, choices=['adam', 'sgd'], default='adam')
	parser.add_argument('-lr', type=float, default=0.0002)
	parser.add_argument('-lr2', type=float, default=0.002)
	parser.add_argument('-weight_decay', type=float, default=5e-5, help='weight decay')
	parser.add_argument('-lr_decay', type=float, default=0)
	parser.add_argument('-lr_step_size', type=int, default=1)
	parser.add_argument('-lr_decay2', type=float, default=0)
	parser.add_argument('-lr_step_size2', type=int, default=100)
	parser.add_argument('-dropout', type=float, default=0.1)
	parser.add_argument('-gcn_dropout', type=float, default=0.2)
	parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')
	parser.add_argument('-window_model', type=str, choices=['deepsea','expecto','danq'], default='expecto')
	parser.add_argument('-loss', type=str, choices=['ce'], default='ce')
	parser.add_argument('-br_threshold', type=float, default=0.5)
	parser.add_argument('-no_cuda', action='store_true')
	parser.add_argument('-shuffle_train', action='store_true')
	parser.add_argument('-pretrain', action='store_true')
	parser.add_argument('-viz', action='store_true')
	parser.add_argument('-gpu_id', type=int, default=-1)
	parser.add_argument('-small', action='store_true')
	parser.add_argument('-summarize_data', action='store_true')
	parser.add_argument('-overwrite', action='store_true')
	parser.add_argument('-test_only', action='store_true')
	parser.add_argument('-load_pretrained', action='store_true')
	parser.add_argument('-seq_length', type=int, default=2000)
	parser.add_argument('-gcn_layers', type=int, default=2)
	parser.add_argument('-save_feats', action='store_true')
	parser.add_argument('-saved_model', type=str, default='')
	parser.add_argument('-A_saliency', action='store_true')
	parser.add_argument('-chrome_model', type=str, choices=['gcn', 'rnn'], default='gcn')
	parser.add_argument('-adj_type', type=str, choices=['constant', 'hic', 'both','random','none',''], default='hic')
	parser.add_argument('-hicnorm', type=str, choices=['KR', 'VC','SQRTVC',''], default='SQRTVC')
	parser.add_argument('-hicsize', type=str, choices=['125000','250000','500000','1000000'], default='1000000')
	parser.add_argument('-gate', action='store_true')
	parser.add_argument('-load_gcn', action='store_true')
	parser.add_argument('-noeye', action='store_true')
	parser.add_argument('-name', type=str, default=None) 
	parser.add_argument('-name2', type=str, default=None) 
	opt = parser.parse_args()
	return opt


def config_args(opt):
	
	if opt.test_batch_size <= 0:
		opt.test_batch_size = opt.batch_size

	opt.graph_root = path.join('/p/qdata/jjl5sw/ChromeGCN/processed_data/',opt.cell_type,opt.window_size,'hic')

	opt.dec_dropout = opt.dropout

	opt.drop_last = True
	if opt.test_only:
		opt.drop_last = False

	opt.model_name = 'graph.'
	opt.model_name += opt.window_model
	opt.model_name += '.'+str(opt.d_model)
	opt.model_name += '.bsz_'+str(opt.batch_size)
	opt.model_name += '.loss_'+str(opt.loss)
	opt.model_name += '.'+str(opt.optim)
	opt.model_name += '.lr_'+str(opt.lr).split('.')[1]

	if opt.lr_decay > 0:
		opt.model_name += '.decay_'+str(opt.lr_decay).replace('.','')+'_'+str(opt.lr_step_size)

	opt.model_name += '.drop_'+("%.2f" % opt.dropout).split('.')[1]+'_'+("%.2f" % opt.dec_dropout).split('.')[1]

	if opt.pretrain:
		print('PRETRAINING')

	if opt.name:
		opt.model_name = (opt.model_name+'.'+str(opt.name))

	if opt.save_feats:
		opt.pretrain = False
		opt.shuffle_train = False
		opt.epochs = 1

	elif opt.load_pretrained:
		opt.model_name += '.finetune'
		opt.model_name += '.lr2_'+str(opt.lr2).split('.')[1]
		opt.model_name += '.gcndrop_'+("%.2f" % opt.gcn_dropout).split('.')[1]
		opt.model_name += '.'+str(opt.optim2)
		opt.model_name += '.'+str(opt.chrome_model)
		opt.model_name += '.layers_'+str(opt.gcn_layers)
		if opt.chrome_model == 'gcn' and opt.gate:
			opt.model_name += '.gate'
		if (opt.chrome_model == 'gcn' or opt.chrome_model == 'ggcn'):
			opt.model_name += '.adj_'+opt.adj_type
			if opt.adj_type == 'hic' or opt.adj_type == 'both':
				opt.model_name += '.norm_'+opt.hicnorm
			if opt.noeye:
				opt.model_name += '.noeye'
		if opt.lr_decay2 > 0:
			opt.model_name += '.decay_'+str(opt.lr_decay2).replace('.','')+'_'+str(opt.lr_step_size2)
			
		if opt.name2 != None:
			opt.model_name += '.'+opt.name2

	opt.model_name = path.join(opt.results_dir,opt.cell_type,opt.model_name)

	opt.dataset = path.join(opt.dataroot,opt.cell_type,opt.window_size)
	opt.cuda = not opt.no_cuda
	opt.d_word_vec = opt.d_model

	if opt.small:
		opt.data = path.join(opt.dataset,'train_valid_test_small.pt')
	else:
		opt.data = path.join(opt.dataset,'train_valid_test.pt')
	
	if opt.load_gcn:
		opt.model_name +='.load_gcn'
	
	if (not opt.viz) and (not opt.overwrite) and (not 'test' in opt.model_name) and (path.exists(opt.model_name)) and (not opt.load_gcn) and (not opt.save_feats):
		print(opt.model_name)
		overwrite_status = input('Already Exists. Overwrite?: ')
		if overwrite_status == 'rm':
			os.system('rm -rf '+opt.model_name)
		elif not 'y' in overwrite_status:
			exit(0)
	
	if not opt.pretrain:
		opt.batch_size = 512
		opt.test_batch_size = 512

	opt.src_vocab_size = 5 #TODO: fix
	
	return opt
