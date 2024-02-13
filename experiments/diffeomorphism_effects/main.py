import argparse
import os
import torch
from torch.utils.data import DataLoader
from datetime import datetime

from data.toy_data.spiral import twospirals

from src.diffeomorphisms.iresnet_euclidean_product import i_ResNet_into_Euclidean
from src.manifolds.pull_back_manifold import PullBackManifold
from src.utils.isomap import make_adjacency
from src.utils.neural_network.distance_data_set import DistanceData


# ------------------------ Start of the main experiment script

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # ------------------------ Input arguments

    # Run parameters
    parser.add_argument('--epochs', type=int, default=20,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size. Does not scale with number of gpus.')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--beta_a', type=float, default=0.9,
                        help='first beta in Adam optimiser')
    parser.add_argument('--beta_b', type=float, default=0.99,
                        help='second beta in Adam optimiser')
    parser.add_argument('--weight_decay', type=float, default=1e-10,
                        help='weight decay')
    parser.add_argument('--log', type=eval, default=True,
                        help='logging flag')
    parser.add_argument('--seed', type=int, default=31,
                        help='Random seed')
    parser.add_argument('--val_interval', type=int, default=5, metavar='N',
                        help='how many epochs to wait before logging validation')

    # Train settings
    parser.add_argument('--exp_no', type=int, default=0,
                        help='experiment number')
    parser.add_argument('--a_sub', type=float, default=10.,
                        help='subspace regularisation parameter')
    parser.add_argument('--a_iso', type=float, default=1e-2,
                        help='isometry regularisation parameter')

    # Spiral Dataset
    parser.add_argument('--num_training_samples', type=int, default=51, metavar='N',
                        help='amount of training samples')
    parser.add_argument('--training_noise_level', type=float, default=0.1,
                        help='variance of the noise added to the training data')
    parser.add_argument('--num_test_samples', type=int, default=501, metavar='N',
                        help='maximum amount of test samples')
    parser.add_argument('--test_noise_level', type=float, default=0.5,
                        help='variance of the noise added to the test data')
    parser.add_argument('--dataset', type=str, default="spiral", metavar='N',
                        help='spiral data set')
    parser.add_argument('--cut_off_constant', type=float, default=75.,
                        help='TODO')

    # iResNet model settings
    parser.add_argument('--nBlocks', type=int, default=100,
                        help='Number of residual blocks')
    parser.add_argument('--int_features', type=int, default=10,
                        help='internal feature dimension')
    parser.add_argument('--coeff', type=float, default=0.8,
                        help='Lipschitz coefficient of a block')
    parser.add_argument('--n_power_iter', type=int, default=10,
                        help='Number of power iterations to compute spectral norm')
    parser.add_argument('--max_iter_inverse', type=int, default=50,
                        help='Number of fixed point iterations for block inversion')

    # Parallel computing stuff
    parser.add_argument('-g', '--gpus', default=0, type=int,
                        help='number of gpus to use (assumes all are on one node)')

    # Arg parser
    args = parser.parse_args()

    torch.manual_seed(31)

    # ------------------------ Device settings

    if args.gpus > 0:  # TODO make sure it works for gpus
        accelerator = "gpu"
        devices = args.gpus
    else:
        accelerator = "cpu"
        devices = "auto"

    # ------------------------ Dataset

    # Load the dataset and set the dataset specific settings
    s_data = twospirals(int((args.num_training_samples -1)/2), noise=args.training_noise_level)

    # Construct similarity graph and compute geodesic distances
    cut_off = args.cut_off_constant / args.num_training_samples  # 1.5 make sure to have full data path
    pairwise_distances = make_adjacency(s_data.T, eps=cut_off)
    pairwise_distances = torch.from_numpy(pairwise_distances)

    dataset_train = DistanceData(s_data, pairwise_distances)

    # Make the dataloaders
    dataloader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)

    # ------------------------ Load and initialize the model

    # construct pull-back manifold
    s_offset = s_data[int((args.num_training_samples - 1) / 2)]
    _, _, s_orthogonal = torch.linalg.svd(s_data)

    s_diffeo = i_ResNet_into_Euclidean([1, 1], s_offset, s_orthogonal,
                                       nBlocks=args.nBlocks,
                                       max_iter_inverse=args.max_iter_inverse,
                                       int_features=args.int_features,
                                       coeff=args.coeff,
                                       n_power_iter=args.n_power_iter)
    s_M = PullBackManifold(s_diffeo)

    # ------------------------ Set up the trainer

    optimizer = torch.optim.Adam(s_M.diffeo.phi.parameters(), lr=args.lr, betas=(args.beta_a, args.beta_b))

    train_with_subspace_regularisation = bool(int('{0:02b}'.format(args.exp_no)[0]))
    train_with_isometry_regularisation = bool(int('{0:02b}'.format(args.exp_no)[1]))
    print(f"start training {'with' if train_with_subspace_regularisation else 'without'} subspace regularisation and {'with' if train_with_isometry_regularisation else 'without'} isometry regularisation")

    average_loss_progression = []
    for epoch in range(args.epochs):
        total_pairwise_distance_loss = 0.0
        total_subspace_loss = 0.0
        total_isometry_loss = 0.0

        for idx, (data_i, data_j, pairwise_distance_ij) in enumerate(dataloader):

            x_i = data_i
            x_j = data_j
            d_ij = pairwise_distance_ij

            phi_i = s_M.diffeo.forward(x_i)
            phi_j = s_M.diffeo.forward(x_j)

            # compute metric tensor at points
            g_i = s_M.metric_tensor_in_std_basis(x_i)
            g_j = s_M.metric_tensor_in_std_basis(x_j)

            # optimize the net
            d_M_ij = s_M.distance(x_i[:, None, :], x_j[:, None, :]).squeeze()
            pairwise_distance_loss = torch.mean((d_M_ij - d_ij) ** 2)  # (\|phi_i - phi_j\| - d_ij)^2

            # subspace loss
            if train_with_subspace_regularisation:
                subspace_loss_i = torch.mean(phi_i[1].norm(1, -1))
                subspace_loss_j = torch.mean(phi_j[1].norm(1, -1))
                subspace_loss = args.a_sub * (subspace_loss_i + subspace_loss_j) / 2
            else:
                subspace_loss = torch.tensor([0.])

            # isometry loss
            if train_with_isometry_regularisation:
                isometry_loss_i = torch.mean(torch.sum((g_i - torch.eye(2)[None]) ** 2, (1, 2)))
                isometry_loss_j = torch.mean(torch.sum((g_j - torch.eye(2)[None]) ** 2, (1, 2)))
                isometry_loss = args.a_iso * (isometry_loss_i + isometry_loss_j) / 2
            else:
                isometry_loss = torch.tensor([0.])

            loss = pairwise_distance_loss + subspace_loss + isometry_loss

            # update networks
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_pairwise_distance_loss += pairwise_distance_loss.item()
            total_subspace_loss += subspace_loss.item()
            total_isometry_loss += isometry_loss.item()

            if (idx % args.val_interval == args.val_interval - 1):
                avg_pairwise_distance_loss = total_pairwise_distance_loss / args.val_interval
                avg_subspace_loss = total_subspace_loss / args.val_interval
                avg_isometry_loss = total_isometry_loss / args.val_interval

                average_loss_progression.append(avg_pairwise_distance_loss + avg_subspace_loss + avg_isometry_loss)
                print(
                    "epoch[{}/{}] mini-batch[{}/{}] avg_pwd_loss {:.6f} avg_subspace_loss {:.6f} avg_isometry_loss {:.6f}" \
                    .format(epoch + 1, args.epochs, idx + 1, len(dataloader), avg_pairwise_distance_loss,
                            avg_subspace_loss, avg_isometry_loss))
                total_pairwise_distance_loss = 0.0
                total_subspace_loss = 0.0
                total_isometry_loss = 0.0

    # ------------------------ Save results

    trained_networks_path = os.path.join("models", "spiral")
    os.makedirs(trained_networks_path, exist_ok=True)

    s_z_path = os.path.join(trained_networks_path, "z.pt")
    s_O_path = os.path.join(trained_networks_path, "O.pt")

    now = datetime.now()
    date_time = now.strftime("%Y_%m_%d_%H_%M_%S")
    s_phi_path = os.path.join(trained_networks_path,
                              "i_resnet_euclidean_product_{}_blocks_{}_int_features_{}_epochs_{}_subspace_reg_{}_isometry_reg_{}.pt".format(
                                  args.nBlocks, args.int_features, args.epochs, train_with_subspace_regularisation,
                                  train_with_isometry_regularisation, date_time))
    s_error_progression_path = os.path.join(trained_networks_path,
                                            "loss_progression_{}_blocks_{}_int_features_{}_epochs_{}_subspace_reg_{}_isometry_reg_{}.pt".format(
                                                args.nBlocks, args.int_features, args.epochs,
                                                train_with_subspace_regularisation,
                                                train_with_isometry_regularisation, date_time))

    # save z, O and phi
    torch.save(s_M.diffeo.phi.state_dict(), s_phi_path)
    torch.save(s_offset, s_z_path)
    torch.save(s_orthogonal, s_O_path)

    # plot loss progression
    torch.save(torch.tensor(average_loss_progression), s_error_progression_path)
