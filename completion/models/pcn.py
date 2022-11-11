from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
import math

# from utils.model_utils import gen_grid_up, calc_emd, calc_cd
#from model_utils import gen_grid_up, calc_emd, calc_cd, gen_jet_corrections, calc_dcd
from model_utils import gen_grid_up, calc_emd, calc_cd,

class PCN_encoder(nn.Module):
    def __init__(self, output_size=75):
        super(PCN_encoder, self).__init__()
        #self.conv1 = nn.Conv1d(4, 8, 1)
        self.conv1 = nn.Conv1d(3, 256, 1)
        self.conv2 = nn.Conv1d(256, 512, 1)
        self.conv3 = nn.Conv1d(1064, 2048, 1)
        self.conv4 = nn.Conv1d(2048, output_size, 1)

    def forward(self, x):
        batch_size, _, num_points = x.size()
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        global_feature, _ = torch.max(x, 2)
        x = torch.cat((x, global_feature.view(batch_size, -1, 1).repeat(1, 1, num_points).contiguous()), 1)
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        global_feature, _ = torch.max(x, 2)
        return global_feature.view(batch_size, -1)


class PCN_decoder(nn.Module):
    def __init__(self, num_coarse, num_fine, scale, cat_feature_num):
        super(PCN_decoder, self).__init__()
        self.num_coarse = num_coarse
        self.num_fine = num_fine
        self.fc1 = nn.Linear(5, 75)
        self.fc2 = nn.Linear(75, 75)
        #self.fc3 = nn.Linear(75, num_coarse * 4)
        self.fc3 = nn.Linear(75, num_coarse * 3)
        
        self.scale = scale
        self.grid = gen_grid_up(2 ** (int(math.log2(scale))), 0.05).cuda().contiguous()
        self.conv1 = nn.Conv1d(cat_feature_num, 2048, 1)
        self.conv2 = nn.Conv1d(2048, 2048, 1)
        #self.conv3 = nn.Conv1d(64, 4, 1)
        self.conv3 = nn.Conv1d(2048, 3, 1)
        
    def forward(self, x):
        batch_size = x.size()[0]
        coarse = F.relu(self.fc1(x))
        coarse = F.relu(self.fc2(coarse))
        #coarse = self.fc3(coarse).view(-1, 4, self.num_coarse)
        coarse = self.fc3(coarse).view(-1, 3, self.num_coarse)

        grid = self.grid.clone().detach()
        grid_feat = grid.unsqueeze(0).repeat(batch_size, 1, self.num_coarse).contiguous().cuda()

        point_feat = (
            (coarse.transpose(1, 2).contiguous()).unsqueeze(2).repeat(1, 1, self.scale, 1).view(-1, self.num_fine,
                                                                                                #4)).transpose(1,
                                                                                                3)).transpose(1,
                                                                                                              2).contiguous()

        global_feat = x.unsqueeze(2).repeat(1, 1, self.num_fine)

        feat = torch.cat((grid_feat, point_feat, global_feat), 1)

        center = ((coarse.transpose(1, 2).contiguous()).unsqueeze(2).repeat(1, 1, self.scale, 1).view(-1, self.num_fine,
                                                                                                      #4)).transpose(1,
                                                                                                      3)).transpose(1,
                                                                                                                    2).contiguous()

        fine = self.conv3(F.relu(self.conv2(F.relu(self.conv1(feat))))) + center
        return coarse, fine


class Model(nn.Module):
    def __init__(self, args, num_coarse=75):
        super(Model, self).__init__()

        self.num_coarse = num_coarse
        self.num_points = args.num_points
        self.train_loss = args.loss
        self.eval_emd = args.eval_emd
        self.scale = self.num_points // num_coarse
        #self.cat_feature_num = 2 + 4 + 75
        self.cat_feature_num = 2 + 3 + 75

        self.encoder = PCN_encoder()
        self.decoder = PCN_decoder(num_coarse, self.num_points, self.scale, self.cat_feature_num)

    def forward(self, x, gt=None, prefix="train", mean_feature=None, alpha=None):
        
        x,mask = gen_jet_corrections(x, ret_mask_separate = True)
        gt,mask_mask = gen_jet_corrections(gt, ret_mask_separate = True)
        
        feat = self.encoder(x)
        out1, out2 = self.decoder(feat)
        out1 = out1.transpose(1, 2).contiguous()
        out2 = out2.transpose(1, 2).contiguous()

        
        a = 0.3
        b = 0.4
        g = 0.3
        
        
        
        if prefix=="train":
            if self.train_loss == 'emd':
                loss1 = calc_emd(out1, gt)
                loss2 = calc_emd(out2, gt)
            elif self.train_loss == 'cd':
                
                lossMSE = nn.MSELoss(reduction = 'none')
                loss1MSE = lossMSE(out1,gt)
                loss2MSE = lossMSE(out2,gt)
                
                # Classic CD
                loss1cd, _ = calc_cd(out1, gt)
                loss2cd, _ = calc_cd(out2, gt)
                
                # DCD
                #loss_opts_alpha = 200
                #loss_opts_lambda = 0.5
                #loss1cd, _, _ = calc_dcd(out1, gt, alpha=loss_opts_alpha, n_lambda=loss_opts_lambda)
                #loss2cd, _, _ = calc_dcd(out2, gt, alpha=loss_opts_alpha, n_lambda=loss_opts_lambda)
                
                #out1pt = out1[:,:,2] # pt
                #out1dis = out1[:,:,-1] # eta,phi

                out2pt = out2[:,:,2] # pt
                out2dis =  out2[:,:,-1] # eta, phi
                
                gtpt = gt[:,:,2] # pt
                gtdis = gt[:,:,-1] # eta,phi
                
                #loss1MSE_pt = lossMSE(out1pt, gtpt)
                loss2MSE_pt = lossMSE(out2pt, gtpt)
                #loss1MSE_dis = lossMSE(out1dis, gtdis)
                loss2MSE_dis = lossMSE(out2dis, gtdis)
                
                loss1MSE_totdis = loss1MSE_pt * loss1MSE_dis
                loss2MSE_totdis = loss2MSE_pt * loss2MSE_dis
                
                #loss2 = (a * loss2MSE) + (b * loss2cd) + (g * loss2MSE_totdis)
                
            else:
                raise NotImplementedError('Train loss is either CD or EMD!')

            total_train_loss_cd = loss1cd.mean() + loss2cd.mean() * alpha  #ATTENZIONE, ALPHA PRE-ESISTENTE
            total_train_loss_MSE = loss1MSE.mean() + loss2MSE.mean()
            total_train_loss_MSE_dis = loss1MSE_totdis.mean() + loss2MSE_totdis.mean()
            total_train_loss = (a * total_train_loss_MSE) + (b*total_train_loss_cd) + (g*total_train_loss_MSE_dis)
            #(total_train_loss = loss_cd * alpha)
            
            return out2, loss2, total_train_loss
        elif prefix=="val":
            if self.eval_emd:
                emd = calc_emd(out2, gt, eps=0.004, iterations=3000)
            else:
                emd = 0
            cd_p, cd_t, f1 = calc_cd(out2, gt, calc_f1=True)
            lossMSE = nn.MSELoss(reduction = 'none')
            loss2MSE = lossMSE(out2,gt)
            
            #loss_opts_alpha = 200
            #loss_opts_lambda = 0.5
            #loss2cd, _, _ = calc_dcd(out2, gt, alpha=loss_opts_alpha, n_lambda=loss_opts_lambda)
           
            out2pt = out2[:,:,2] # pt
            out2dis =  out2[:,:,-1] # eta, phi
                
            gtpt = gt[:,:,2] # pt
            gtdis = gt[:,:,-1] # eta,phi
            loss2MSE_pt = lossMSE(out2pt, gtpt)
            loss2MSE_dis = lossMSE(out2dis, gtdis)
            loss2MSE_totdis = loss2MSE_pt * loss2MSE_dis
            a = 0.3
            b = 0.4
            g = 0.3
            loss2 = (a * loss2MSE) + (b * loss2cd) + (g * loss2MSE_totdis)
            return {'out1': out1, 'out2': out2, 'emd': emd, 'cd_p': cd_p, 'cd_t': cd_t, 'f1': f1, 'tot_loss': loss2}
        else:
            return {'result': out2}
