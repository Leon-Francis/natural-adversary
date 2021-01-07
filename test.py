import torch
from models import Seq2SeqCAE
from transformers import BertTokenizer

autoencoder = Seq2SeqCAE(emsize=args.emsize,
                                     nhidden=args.nhidden,
                                     ntokens=ntokens,
                                     nlayers=args.nlayers,
                                     noise_radius=args.noise_radius,
                                     hidden_init=args.hidden_init,
                                     dropout=args.dropout,
                                     conv_layer=args.arch_conv_filters,
                                     conv_windows=args.arch_conv_windows,
                                     conv_strides=args.arch_conv_strides,
                                     gpu=args.cuda)

autoencoder.load_state_dict(torch.load('output/1605104206/models/autoencoder_model.pt'))
train_data = 'Fears for T N pension after talks. Unions representing workers at Turner   Newall say they are \'disappointed\' after talks with stricken parent firm Federal Mogul.'
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
data_tokens = tokenizer.tokenize(train_data)[:9] + ['[SEP]']
label_tokens = ['CLS'] + tokenizer.tokenize(train_data)[:9]
data_idx = []
label_idx = []

data_idx.append(tokenizer.convert_tokens_to_ids(data_tokens))
label_idx.append(tokenizer.convert_tokens_to_ids(label_tokens))
data_idx = torch.tensor(data_idx)
label_idx = torch.tensor(label_idx)
hidden = autoencoder()