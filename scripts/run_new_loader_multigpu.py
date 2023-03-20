import os
import numpy as np
import warnings
from collections import defaultdict
from numpy.lib.function_base import append
from sklearn.utils import random
from sklearn.utils.random import sample_without_replacement

import torch
from torch.optim import Adam
import yaml
from newtonnet.layers.activations import get_activation_by_string
from newtonnet.models import NewtonNet
from torch.utils.data import DataLoader

from newtonnet.train import Trainer
# from newtonnet.data.parse_raw import *
# from newtonnet.data import parse_train_test, parse_ani_ccx_data#, new_parser,BatchDataset
from newtonnet.data.parse_raw import parse_new
from newtonnet.data.loader import BatchDataset, collate_wrapper
import argparse
import torch.distributed as dist

parser = argparse.ArgumentParser(description='NANET Training')
parser.add_argument("--local_rank", metavar="Local Rank", type=int, default=0, 
                    help="Torch distributed will automatically pass local argument")
parser.add_argument("--cfg", metavar="Config Filename", default='config_disable_newton_option_dgx.yml', 
                    help="Experiment to run. Default is Imagenet 300 epochs")
parser.add_argument("--name", metavar="Log Name", default="", 
                    help="Name of wandb entry")
parser.add_argument("--skip_eval", action='store_true',
                    help="Name of wandb entry")
parser.add_argument("--p", default='ani_ccx',
                    help="parser")
parser.add_argument("--print_freq", default=20,type=int,
                    help="Log Printing Frequency")

parser.add_argument("--resume_path", default=None,type=str,
                    help="Resume path")


# from newtonnet.loader import BatchDataset
# torch.autograd.set_detect_anomaly(True)
def main(args):
    torch.set_default_tensor_type(torch.DoubleTensor)

    # settings
    settings_path = args.cfg 
    settings = yaml.safe_load(open(settings_path, "r"))


    # device
    config = {}
    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))        
    config.update({'world_size': world_size, 'rank': rank, 'local_rank': local_rank})

    dist.init_process_group(backend="nccl", world_size=world_size, rank=rank)
    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)
  
    print("RUN DEVICE", device)
    # data
    # train_gen, val_gen, test_gen, tr_steps, val_steps, test_steps, n_tr_data, n_val_data, n_test_data, normalizer, test_energy_hash = parse_ani_ccx_data(settings,'original',device)
    # TODO: FIX LOGIC FOR ALL THESE
    parsed_data = None
    if args.p in ['ccsd']:
        train_mode = 'energy/force'
    elif args.p in ['ani']:
        train_mode = 'energy'
        parsed_data = parse_new(settings, 'original', device, "ani", "ani") 
        print('data set:', 'ANI')
    elif args.train in ['ani_ccx']:
        train_mode = 'energy'
        parsed_data = parse_new(settings, 'original', device, "ani_ccx", "ani_ccx") 
        print('data set:', 'ANI_CCX')
    elif args.p in ['md17']:
        train_mode = 'energy'
        parsed_data = parse_new(settings, 'original', device, "ani_ccx", "md17") 
        print('data set:', 'MD17 for Test and Ani-cxx for Train')
    elif args.p in ['methane']:
        train_mode = 'energy/force'
        print('data set:', 'Methane Combustion')
    elif args.p in ['hydrogen']:
        train_mode = 'energy/force'
        print('data set:', 'Methane Combustion')
    else:
        raise NotImplementedError
    
    if parsed_data is None:
        raise NotImplementedError
    
    #TODO: FIX ALL ABOVE

    dtrain, dval, dtest, env, tr_steps, val_steps, test_steps, n_tr_data, n_val_data, n_test_data, normalizer, test_energy_hash, tr_batch_size, tr_rotations, val_batch_size, val_rotations = parsed_data
    train_batch_dataset = BatchDataset(dtrain, 'cpu') # should send data to cpu always
    val_batch_dataset = BatchDataset(dval, 'cpu')
    test_batch_dataset = BatchDataset(dtest, 'cpu')
    sampler_train = torch.utils.data.DistributedSampler(
            train_batch_dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    sampler_val = torch.utils.data.DistributedSampler(
            val_batch_dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    sampler_test = torch.utils.data.DistributedSampler(
            test_batch_dataset, num_replicas=world_size, rank=rank, shuffle=True
    )

    train_gen = DataLoader(train_batch_dataset, batch_size=tr_batch_size, 
                                collate_fn=collate_wrapper(env_provider=env,
                                            n_rotations=tr_rotations,
                                            freeze_rotations=settings['training']['tr_frz_rot'],
                                            keep_original=settings['training']['tr_keep_original'],
                                            device=device,
                                            shuffle=settings['training']['shuffle'],
                                            drop_last=settings['training']['drop_last']),sampler=sampler_train)

    
    val_gen = DataLoader(val_batch_dataset, batch_size=tr_batch_size, 
                                collate_fn=collate_wrapper(env_provider=env,
                                            n_rotations=tr_rotations,
                                            freeze_rotations=settings['training']['tr_frz_rot'],
                                            keep_original=settings['training']['tr_keep_original'],
                                            device=device,
                                            shuffle=settings['training']['shuffle'],
                                            drop_last=settings['training']['drop_last']),
                                        sampler=sampler_val)

    
    test_gen = DataLoader(test_batch_dataset, batch_size=tr_batch_size, sampler=sampler_test,
                                collate_fn=collate_wrapper(env_provider=env,
                                            n_rotations=tr_rotations,
                                            freeze_rotations=settings['training']['tr_frz_rot'],
                                            keep_original=settings['training']['tr_keep_original'],
                                            device=device,
                                            shuffle=settings['training']['shuffle'],
                                            drop_last=settings['training']['drop_last']))                    

    # model
    # activation function
    activation = get_activation_by_string(settings['model']['activation'])

    model = NewtonNet(resolution=settings['model']['resolution'],
                n_features=settings['model']['n_features'],
                activation=activation,
                n_interactions=settings['model']['n_interactions'],
                dropout=settings['training']['dropout'],
                max_z=10,
                cutoff=settings['data']['cutoff'],  ## data cutoff
                cutoff_network=settings['model']['cutoff_network'],
                normalizer=normalizer,
                normalize_atomic=settings['model']['normalize_atomic'],
                requires_dr=settings['model']['requires_dr'],
                device=device,
                create_graph=True,
                shared_interactions=settings['model']['shared_interactions'],
                return_latent=settings['model']['return_latent'],
                double_update_latent=settings['model']['double_update_latent'],
                layer_norm=settings['model']['layer_norm'],
                aggregration=settings['model']['aggregration'],
                nonlinear_attention = settings['model']['nonlinear_attention'],
                newtonian_dynamics=settings['model']['newtonian_dynamics']
                )
    model = model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model,find_unused_parameters=True)
    model_without_ddp = model.module
    # laod pre-trained model
    if settings['model']['pre_trained']:
        model_path = settings['model']['pre_trained']
        model.load_state_dict(torch.load(model_path)['model_state_dict'])

    # optimizer
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = Adam(trainable_params,
                    lr=settings['training']['lr'],
                    weight_decay=settings['training']['weight_decay'])

    # loss
    w_energy = settings['model']['w_energy']
    w_force = settings['model']['w_force']
    w_f_mag = settings['model']['w_f_mag']
    w_f_dir = settings['model']['w_f_dir']

    def custom_loss(preds, batch_data, w_e=w_energy, w_f=w_force, w_fm=w_f_mag, w_fd=w_f_dir):

        # compute the mean squared error on the energies
        # print(batch_data.keys())
        diff_energy = preds['E'] - batch_data["E"].view(preds['E'].shape)
        # print("diff_energy.shape", preds['E'].shape,  batch_data['E'].shape, diff_energy.shape)
        assert diff_energy.shape[1] == 1
        err_sq_energy = torch.mean(diff_energy**2)
        err_sq = w_e * err_sq_energy

        # compute the mean squared error on the forces
        # print(preds['F'].shape,batch_data["F"].shape)
        diff_forces = preds['F'] - batch_data["F"]
        err_sq_forces = torch.mean(diff_forces**2)
        err_sq = err_sq + w_f * err_sq_forces

        # compute the mean square error on the force magnitudes
        if w_fm > 0:
            diff_forces = torch.norm(preds['F'], p=2, dim=-1) - torch.norm(batch_data["F"], p=2, dim=-1)
            err_sq_mag_forces = torch.mean(diff_forces ** 2)
            err_sq = err_sq + w_fm * err_sq_mag_forces

        if w_fd > 0:
            cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
            direction_diff = 1 - cos(preds['hs'][-1][1], batch_data["F"])
            # direction_diff = direction_diff * torch.norm(batch_data["F"], p=2, dim=-1)
            direction_loss = torch.mean(direction_diff)
            err_sq = err_sq + w_fd * direction_loss

        if settings['checkpoint']['verbose']:
            print('\n',
                ' '*8, 'energy loss: ', err_sq_energy.detach().cpu().numpy(), '\n')
            print(' '*8, 'force loss: ', err_sq_forces.detach().cpu().numpy(), '\n')

            if w_fm>0:
                print(' '*8, 'force mag loss: ', err_sq_mag_forces, '\n')

            if w_fd>0:
                print(' '*8, 'direction loss: ', direction_loss.detach().cpu().numpy())

        return err_sq


    # training
    trainer = Trainer(model=model,
                    loss_fn=custom_loss,
                    optimizer=optimizer,
                    requires_dr=settings['model']['requires_dr'],
                    device=device,
                    yml_path=settings['general']['me'],
                    output_path=settings['general']['output'],
                    script_name=settings['general']['driver'],
                    lr_scheduler=settings['training']['lr_scheduler'],
                    energy_loss_w= w_energy,
                    force_loss_w=w_force,
                    loss_wf_decay=settings['model']['wf_decay'],
                    checkpoint_log=settings['checkpoint']['log'],
                    checkpoint_val=settings['checkpoint']['val'],
                    checkpoint_test=settings['checkpoint']['test'],
                    checkpoint_model=settings['checkpoint']['model'],
                    verbose=settings['checkpoint']['verbose'],
                    hooks=settings['hooks'],
                    mode=train_mode,
                    print_freq=args.print_freq)
    
    if args.resume_path:
        trainer.resume_model(args.resume_path)

    trainer.print_layers()

    # tr_steps=1; val_steps=0; irc_steps=0; test_steps=0
    print(f"DEBUG: START TRAINING {local_rank,device}")
    trainer.train(train_generator=train_gen,
                epochs=settings['training']['epochs'],
                steps=tr_steps,
                val_generator=val_gen,
                val_steps=val_steps,
                irc_generator=None,
                irc_steps=None,
                test_generator=test_gen,
                test_steps=test_steps,
                clip_grad=1.0)

    print('done!')


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)