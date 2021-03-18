# Hamiltonian Neural Networks | 2019
# Sam Greydanus, Misko Dzamba, Jason Yosinski

import torch, argparse
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import scipy

import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

from nn_models import MLPAutoencoder
from hnn import HNN
from data import get_dataset
from utils import L2_loss, to_pickle, from_pickle, give_min_and_dist, scale, unscale

def get_args():
  parser = argparse.ArgumentParser(description=None)
  parser.add_argument('--input_dim', default=4, type=int, help='dimensionality of input tensor')
  parser.add_argument('--hidden_dim', default=200, type=int, help='hidden dimension of mlp')
  parser.add_argument('--latent_dim', default=4, type=int, help='latent dimension of mlp')
  parser.add_argument('--learn_rate', default=1e-5, type=float, help='learning rate')
  parser.add_argument('--batch_size', default=200, type=int, help='batch_size')
  parser.add_argument('--input_noise', default=0.0, type=int, help='std of noise added to inputs')
  parser.add_argument('--nonlinearity', default='tanh', type=str, help='neural net nonlinearity')
  parser.add_argument('--total_steps', default=2, type=int, help='number of gradient steps')
  parser.add_argument('--print_every', default=200, type=int, help='number of gradient steps between prints')
  parser.add_argument('--name', default='2body', type=str, help='only one option right now')
  parser.add_argument('--baseline', dest='baseline', action='store_true', help='run baseline or experiment?')
  parser.add_argument('--verbose', dest='verbose', action='store_true', help='verbose?')
  parser.add_argument('--field_type', default='solenoidal', type=str, help='type of vector field to learn')
  parser.add_argument('--seed', default=0, type=int, help='random seed')
  parser.add_argument('--save_dir', default=THIS_DIR, type=str, help='where to save the trained model')
  parser.set_defaults(feature=True)
  return parser.parse_args()

def train(args):
  # set random seed
  torch.manual_seed(args.seed)
  np.random.seed(args.seed)
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print(device)
  # init model and optimizer
  if args.verbose:
    print("Training baseline model:" if args.baseline else "Training HNN model:")

  output_dim = args.input_dim if args.baseline else 2
  nn_model = MLPAutoencoder(args.input_dim, args.hidden_dim, args.latent_dim, args.nonlinearity)
  nn_model.to(device)
  model = HNN(args.input_dim, differentiable_model=nn_model,
            field_type=args.field_type, baseline=args.baseline, device=device)
  model.to(device)
  optim = torch.optim.Adam(model.parameters(), args.learn_rate, weight_decay=0)
  
  # arrange data
  X = np.load('statrectinputs.npy')
  Y = np.load('statrectoutputs.npy')
  Y[~np.isfinite(Y)] = 0
  xm, xd = give_min_and_dist(X)
  ym, yd= give_min_and_dist(Y)
  X = scale(X, xm, xd)
  Y = scale(Y, ym, yd)
  n_egs = X.shape[0]
  x = X[0:int(0.8*n_egs),:]
  test_x = torch.tensor(X[:-int(0.2*n_egs),:], requires_grad=True, dtype=torch.float32)
  dxdt = Y[0:int(0.8*n_egs),:]
  test_dxdt = torch.tensor(Y[:-int(0.2*n_egs),:])


  # vanilla train loop
  stats = {'train_loss': [], 'test_loss': []}
  for step in range(args.total_steps+1):

    # train step
    ixs = torch.randperm(x.shape[0])[:args.batch_size]
    x = torch.tensor(x[ixs], requires_grad=True, dtype=torch.float32)
    x.to(device)
    dxdt_hat = model.time_derivative(x)
    y = torch.tensor(dxdt[ixs])
    y.to(device)
    loss = L2_loss(y, dxdt_hat)
    loss.backward()
    grad = torch.cat([p.grad.flatten() for p in model.parameters()]).clone()
    optim.step() ; optim.zero_grad()

    # run test data
    test_ixs = torch.randperm(test_x.shape[0])[:args.batch_size]
    test_dxdt_hat = model.time_derivative(test_x[test_ixs])

    #test_dxdt_hat += args.input_noise * torch.randn(*test_x[test_ixs].shape) # add noise, maybe
    test_loss = L2_loss(test_dxdt[test_ixs], test_dxdt_hat)

    # logging
    stats['train_loss'].append(loss.item())
    stats['test_loss'].append(test_loss.item())
    if args.verbose and step % args.print_every == 0:
      print("step {}, train_loss {:.4e}, test_loss {:.4e}, grad norm {:.4e}, grad std {:.4e}"
          .format(step, loss.item(), test_loss.item(), grad@grad, grad.std()))
  ixs = torch.randperm(x.shape[0])[:10000]
  x = torch.tensor(x[ixs], requires_grad=True, dtype=torch.float32)
  x.to(device)
  enc = model.encoding(x).detach().numpy()
  print(x.shape)
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  x = x.detach().numpy()
  img = ax.scatter(enc[:,0], enc[:,3], enc[:,2], c=enc[:,1], cmap=plt.hot())
  fig.colorbar(img)
  plt.savefig('lrep.png')
  y0 = torch.tensor([0.4, 0.3, 1/np.sqrt(2), 1/np.sqrt(2)], requires_grad=True, dtype=torch.float32)
  orbit = sample_orbit(model.time_derivative, [0,10], y0)
  print(orbit.y)
  plt.scatter(orbit.y[0], orbit.y[1])
  plt.savefig('orbit.png')

  return model,  stats

def sample_orbit(model, t_eval, y0):
  t_span = [t_eval[0], t_eval[-1]]
  solution = scipy.integrate.solve_ivp(model, t_span, y0)
  return solution.y


if __name__ == "__main__":
  args = get_args()
  model, stats = train(args)


  # save
  os.makedirs(args.save_dir) if not os.path.exists(args.save_dir) else None
  label = 'baseline' if args.baseline else 'hnn'
  path = '{}/{}-orbits-{}.tar'.format(args.save_dir, args.name, label)
  torch.save(model.state_dict(), path)

