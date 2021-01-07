import torch
from models import Seq2SeqCAE, MLP_G, MLP_I_AE, MLP_D
from train import parse_args

ntokens = 11004
args = parse_args()
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
inverter = MLP_I_AE(ninput=args.nhidden, noutput=args.z_size, layers=args.arch_i)
gan_gen = MLP_G(ninput=args.z_size, noutput=args.nhidden, layers=args.arch_g)
gan_disc = MLP_D(ninput=args.nhidden, noutput=1, layers=args.arch_d)
autoencoder.load_state_dict(torch.load("output/1605104206/models/autoencoder_model.pt"))
inverter.load_state_dict(torch.load("output/1605104206/models/inverter_model.pt"))
gan_gen.load_state_dict(torch.load("output/1605104206/models/gan_gen_model.pt"))
gan_disc.load_state_dict(torch.load("output/1605104206/models/gan_disc_model.pt"))

hypothesis = torch.tensor([1, 8342, 4030, 639, 9883, 5471, 534, 9754, 5549, 8342, 4030, 639, 9883, 5471, 534, 9754, 5549, 8342, 4030, 639, 9883, 5471, 534, 9754, 5549, 8342, 4030, 639, 9883, 5471, 534, 9754, 5549, 8342, 4030, 639, 9883, 5471, 534, 9754, 5549, 22], dtype=torch.long)
lengths = hypothesis.size()
c = autoencoder.encode(hypothesis, lengths, noise=False)
z = inverter(c).data.cpu()
max_indices = autoencoder.generate(gan_gen(z), lengths)

