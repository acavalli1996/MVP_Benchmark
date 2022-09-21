import logging
import os
import sys
import importlib
import argparse
import numpy as np
import h5py
import subprocess

import plotting

from numpy.lib.index_tricks import AxisConcatenator
import munch
import yaml
# from utils.vis_utils import plot_single_pcd
# from utils.train_utils import *
from vis_utils import plot_single_pcd
from train_utils import *
from dataset import MVP_CP

import warnings
warnings.filterwarnings("ignore")


def test():
    dataset_test = MVP_CP(prefix="test")
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size,
                                                  shuffle=False, num_workers=int(args.workers))
    dataset_length = len(dataset_test)
    logging.info('Length of test dataset:%d', len(dataset_test))

    # load model
    model_module = importlib.import_module('.%s' % args.model_name, 'models')
    net = torch.nn.DataParallel(model_module.Model(args))
    net.cuda()
    net.module.load_state_dict(torch.load(args.load_model)['net_state_dict'])
    logging.info("%s's previous weights loaded." % args.model_name)
    net.eval()

    logging.info('Testing...')
    with torch.no_grad():
        results_list = []
        for i, data in enumerate(dataloader_test):
            
            inputs_cpu = data

            inputs = inputs_cpu.float().cuda()
            inputs = inputs.transpose(2, 1).contiguous()

            result_dict = net(inputs, prefix="test")
            results_list.append(result_dict['result'].cpu().numpy())

            if i % args.step_interval_to_print == 0:
                logging.info('test [%d/%d]' % (i, dataset_length / args.batch_size))

        all_results = np.concatenate(results_list, axis=0)
        print(all_results.shape)

        with h5py.File(log_dir + '/results.h5', 'w') as f:
            f.create_dataset('results', data=all_results)
        
        cur_dir = os.getcwd()
        cmd = "cd %s; zip -r submission.zip results.h5 ; cd %s" % (log_dir, cur_dir)
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        _, _ = process.communicate()
        print("Submission file has been saved to %s/submission.zip" % (log_dir))

        
def make_plots(
    #losses,
    #epoch,
    #real_jets,
    gen_jets,
    #real_mask,
    gen_mask,
    #jet_type,
    num_particles,
    name,
    figs_path,
    losses_path,
    save_epochs=5,
    const_ylim=False,
    coords="polarrel",
    dataset="jetnet",
    loss="ls",
):
    """Plot histograms, jet images, loss curves, and evaluation curves"""
    #real_masses = jetnet.utils.jet_features(real_jets)["mass"]
    gen_masses = jetnet.utils.jet_features(gen_jets)["mass"]

#    if "w1efp" in losses:
#        real_efps = jetnet.utils.efps(real_jets)
#        gen_efps = jetnet.utils.efps(gen_jets)
#
#        plotting.plot_part_feats(
#            jet_type,
#            real_jets,
#            gen_jets,
#            real_mask,
#            gen_mask,
#            name=name + "p",
#            figs_path=figs_path,
#            losses=losses,
#            num_particles=num_particles,
#            coords=coords,
#            dataset=dataset,
#            const_ylim=const_ylim,
#            show=False,
#        )
#        plotting.plot_jet_feats(
#            jet_type,
#            real_masses,
#            gen_masses,
#            real_efps,
#            gen_efps,
#            name=name + "j",
#            figs_path=figs_ph,
#            losses=losses,
#            show=False,
#        )
#   else:
     plotting.plot_part_feats_jet_mass(
          #jet_type,
          #real_jets,
          gen_jets,
          #real_mask,
          gen_mask,
          #real_masses,
          gen_masses,
          name=None,
          figs_path=None,
          losses=None,
          num_particles=150,
          coords="polarrel",
          dataset="jetnet",
          const_ylim=False,
          show=False,
      )

#    if len(losses["G"]) > 1:
#        plotting.plot_losses(losses, loss=loss, name=name, losses_path=losses_path, show=False)

#        try:
#            remove(losses_path + "/" + str(epoch - save_epochs) + ".pdf")
#        except:
#            logging.info("Couldn't remove previous loss curves")

#    if len(losses["w1p"]) > 1:
#        plotting.plot_eval(
#            losses,
#            epoch,
#            save_epochs,
#            coords=coords,
#            name=name + "_eval",
#            losses_path=losses_path,
#            show=False,
#        )

#        try:
#            remove(losses_path + "/" + str(epoch - save_epochs) + "_eval.pdf")
#        except:
#            logging.info("Couldn't remove previous eval curves")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test config file')
    parser.add_argument('-c', '--config', help='path to config file', required=True)
    arg = parser.parse_args()
    config_path = arg.config
    args = munch.munchify(yaml.safe_load(open(config_path)))

    if not args.load_model:
        raise ValueError('Model path must be provided to load model!')

    exp_name = os.path.basename(args.load_model)
    log_dir = os.path.dirname(args.load_model)
    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(os.path.join(log_dir, 'test.log')),
                                                      logging.StreamHandler(sys.stdout)])

    test()
    make_plots(
        #losses,
        #epoch,
        #real_jets,
        gen_jets,
        #real_mask,
        gen_mask,
        args.jets,
        args.num_hits,
        str(epoch),
        args.figs_path,
        args.losses_path,
        save_epochs=args.save_epochs,
        const_ylim=args.const_ylim,
        coords=args.coords,
        loss=args.loss,
    )
