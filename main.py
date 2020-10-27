from pdb import set_trace as stop
import argparse,math,time,warnings,copy, pickle, numpy as np, os.path as path 
import torch, torch.nn as nn, torch.nn.functional as F
from models.WindowModels import DeepSEA,Expecto,DanQ
from models.ChromeModels import ChromeGCN,ChromeRNN
from models.NonStrandSpecific import GraphNonStrandSpecific
from data_loader import process_data
from config_args import config_args,get_args
from pdb import set_trace as stop
from utils.evals import Logger
from utils import util_methods
from runner import run_model

warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser()
args = get_args(parser)
opt = config_args(args)


def main(opt):
	###################### Loading Dataset#####################
	print(opt.model_name)

	print('==> Loading Data')
	print(opt.data)
	data = torch.load(opt.data)
	
	print('==> Processing Data')
	if not opt.pretrain and not opt.save_feats:
		train_data = torch.load(opt.model_name.split('.finetune')[0]+'/chrom_feature_dict_train.pt')
		valid_data = torch.load(opt.model_name.split('.finetune')[0]+'/chrom_feature_dict_valid.pt')
		test_data = torch.load(opt.model_name.split('.finetune')[0]+'/chrom_feature_dict_test.pt')
	else:
		train_data,valid_data,test_data = process_data(data,opt)
	opt.tgt_vocab_size = len(data['dict']['tgt'])
	data_dict = data['dict']

	print('==> Creating window_model')
	#################### Creating WindowModel  ####################
	if opt.window_model == 'deepsea':
		WindowBase = DeepSEA(opt.tgt_vocab_size, opt.seq_length)
	elif opt.window_model == 'expecto':
		WindowBase = Expecto(opt.tgt_vocab_size, opt.seq_length)
	elif opt.window_model == 'danq':
		WindowBase = DanQ(opt.tgt_vocab_size)

	WindowModel = GraphNonStrandSpecific(WindowBase)
	# print(WindowModel)

	opt.total_num_parameters = int(util_methods.count_parameters(WindowModel))
	print(opt.total_num_parameters)

	optimizer = util_methods.get_optimizer(WindowModel,opt)
	
	ChromeModel = None
	if not opt.pretrain:
		#################### Creating GNNModel ####################

		if opt.chrome_model == 'rnn':
			ChromeModel = ChromeRNN(128, 128, opt.tgt_vocab_size, opt.gcn_dropout,opt.gcn_layers)
		else:
			ChromeModel = ChromeGCN(128, 128, opt.tgt_vocab_size, opt.gcn_dropout, opt.gate,opt.gcn_layers)
		
		print(ChromeModel)

		if opt.load_gcn:
			print('Loading Saved GCN')
			checkpoint = torch.load(opt.model_name.replace('.load_gcn','')+'/model.chkpt')
			ChromeModel.load_state_dict(checkpoint['model'])
		else:
			# Initialize GCN output layer with window_model output layer
			print('Loading Saved window_model')
			# checkpoint = torch.load(opt.saved_model)
			checkpoint = torch.load(opt.model_name.split('.finetune')[0]+'/model.chkpt')
			WindowModel = nn.DataParallel(WindowModel)
			WindowModel = WindowModel.cuda()
			WindowModel.load_state_dict(checkpoint['model'])
			ChromeModel.out.weight.data = WindowModel.module.model.classifier.weight.data
			ChromeModel.out.bias.data = WindowModel.module.model.classifier.bias.data
			ChromeModel.batch_norm.weight.data = WindowModel.module.model.batch_norm.weight.data
			ChromeModel.batch_norm.bias.data = WindowModel.module.model.batch_norm.bias.data

		optimizer = util_methods.get_optimizer(ChromeModel,opt)


	scheduler = torch.torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
	# scheduler = torch.torch.optim.lr_schedulerReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True,factor=0.8)
	print(optimizer)

	crit = util_methods.get_criterion(opt)

	if torch.cuda.device_count() > 0 and opt.pretrain:
		print("Using", torch.cuda.device_count(), "GPUs!")
		WindowModel = nn.DataParallel(WindowModel)

	if torch.cuda.is_available() and opt.cuda:
		crit = crit.cuda()
		if opt.pretrain: 
			WindowModel = WindowModel.cuda()
		if ChromeModel is not None:
			ChromeModel = ChromeModel.cuda()
		if opt.gpu_id != -1:
			torch.cuda.set_device(opt.gpu_id)

	logger = Logger(opt)

	try:
		run_model(WindowModel,ChromeModel,train_data,valid_data,test_data,
				  crit,optimizer,scheduler,opt,data_dict,logger)
	except KeyboardInterrupt:
		print('-' * 89+'\nManual Exit')
		exit()

if __name__ == '__main__':
	main(opt)
