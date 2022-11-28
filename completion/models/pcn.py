from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
import math
import numpy as np

# from utils.model_utils import gen_grid_up, calc_emd, calc_cd
from model_utils import gen_grid_up, calc_emd, calc_cd


class PCN_encoder(nn.Module):
    def __init__(self, output_size=75):
        super(PCN_encoder, self).__init__()
        #self.conv1 = nn.Conv1d(4, 8, 1)
        self.conv1 = nn.Conv1d(3, 256, 1)
        self.conv2 = nn.Conv1d(256, 512, 1)
        self.conv3 = nn.Conv1d(1024, 2048, 1)
        self.conv4 = nn.Conv1d(2048, output_size, 1)

    def forward(self, x):
        batch_size, _, num_points = x.size()
        #print("1", x.shape)
        x = F.relu(self.conv1(x))
        if torch.isnan(x).any():
          print("x after conv1 è nan")
        #print("2", x.shape)
        x = self.conv2(x)
        if torch.isnan(x).any():
          print("x after conv2 è nan")
        #print("3", x.shape)
        global_feature, _ = torch.max(x, 2)
        if torch.isnan(global_feature).any():
          print("global feature after torch max è nan")
        #print("4", global_feature.shape)
        x = torch.cat((x, global_feature.view(batch_size, -1, 1).repeat(1, 1, num_points).contiguous()), 1)
        if torch.isnan(x).any():
          print("x after torchcat è nan")
        #print("5", x.shape)
        x = F.relu(self.conv3(x))
        if torch.isnan(x).any():
          print("x after conv3 è nan")
        #print("6", x.shape)
        x = self.conv4(x)
        if torch.isnan(x).any():
          print("x after conv4 è nan")
        #print("7", x.shape)
        global_feature, _ = torch.max(x, 2)
        #print("8", global_feature.shape)
        if torch.isnan(global_feature).any():
          print("global feature è nan")
        return global_feature.view(batch_size, -1)


class PCN_decoder(nn.Module):
    def __init__(self, num_coarse, num_fine, scale, cat_feature_num):
        super(PCN_decoder, self).__init__()
        self.num_coarse = num_coarse
        self.num_fine = num_fine
        self.fc1 = nn.Linear(75, 75)
        self.fc2 = nn.Linear(75, 75)
        #self.fc3 = nn.Linear(75, num_coarse * 4)
        self.fc3 = nn.Linear(75, num_coarse * 3)
        
        self.scale = scale
        self.grid = gen_grid_up(2 ** (int(math.log2(scale))), 0.05).cuda().contiguous()
        self.conv1 = nn.Conv1d(cat_feature_num, 2048, 1)
        self.conv2 = nn.Conv1d(2048, 2048, 1)
        #self.conv3 = nn.Conv1d(64, 4, 1)
        self.conv3 = nn.Conv1d(2048, 3, 1)
        
    def forward(self, x, mask):
        #print("10", x.shape)
        batch_size = x.size()[0]
        #print("11 bs", batch_size)
        coarse = F.relu(self.fc1(x))
        #print("12", coarse.shape)
        coarse = F.relu(self.fc2(coarse))
        #print("13",coarse.shape)
        #coarse = self.fc3(coarse).view(-1, 4, self.num_coarse)
        coarse = self.fc3(coarse).view(-1, 3, self.num_coarse)
        #print("14", coarse.shape)

        

        grid = self.grid.clone().detach()
        #print("15", grid.shape)

        # Grid Feat
        grid_feat = grid.unsqueeze(0).repeat(batch_size, 1, self.num_coarse).contiguous().cuda()
        #print("1600", grid_feat.shape)
        s = grid_feat.shape[0]
        grid_feat = torch.reshape(grid_feat,[s,150,2])
        #print("1601", grid_feat.shape)
        #grid_feat[~mask] = 0
        #print("16", grid_feat.__getitem__(0))
        grid_feat = torch.reshape(grid_feat,[s,2,150])

        if torch.isnan(grid_feat).any():
          print("grid_feat è nan")

        # Point feat
        point_feat = (
            (coarse.transpose(1, 2).contiguous()).unsqueeze(2).repeat(1, 1, self.scale, 1).view(-1, self.num_fine,
                                                                                                #4)).transpose(1,
                                                                                                3)).transpose(1,
                                                                                                              2).contiguous()
        s = point_feat.shape[0]
        point_feat = torch.reshape(point_feat,[s,150,3])
        #point_feat[~mask] = 0
        point_feat = torch.reshape(point_feat,[s,3,150])
        #print("17", point_feat.shape)

        if torch.isnan(point_feat).any():
          print("point_feat è nan")

        # Global Feat
        global_feat = x.unsqueeze(2).repeat(1, 1, self.num_fine)
        s = global_feat.shape[0]
        global_feat = torch.reshape(global_feat,[s,150,75])
        #global_feat[~mask] = 0
        global_feat = torch.reshape(global_feat,[s,75,150])
        #print("18", global_feat.shape)

        if torch.isnan(global_feat).any():
          print("global_feat è nan")

        # Feat
        feat = torch.cat((grid_feat, point_feat, global_feat), 1)
        s = grid_feat.shape[0]
        feat = torch.reshape(feat,[s,150,80])
        feat[~mask] = 0
        feat = torch.reshape(feat,[s,80,150])
        #print("19", feat.shape)
        if torch.isnan(feat).any():
          print("feat è nan")


        center = ((coarse.transpose(1, 2).contiguous()).unsqueeze(2).repeat(1, 1, self.scale, 1).view(-1, self.num_fine,
                                                                                                      #4)).transpose(1,
                                                                                                      3)).transpose(1,
                                                                                                                    2).contiguous()
        #print("20", center.shape)
        fine = F.relu(self.conv1(feat))
        #fine = torch.reshape(fine,[256,150,2048])
        #fine[~mask_gt] = 0
        #fine = torch.reshape(fine,[256,2048,150])
        #print("211",fine.shape)


        fine = F.relu(self.conv2(fine))
        #fine = torch.reshape(fine,[256,150,2048])
        #fine[~mask_gt] = 0
        #fine = torch.reshape(fine,[256,2048,150])
        #print("212",fine.shape)


        fine = self.conv3(fine)
        s = fine.shape[0]
        fine = torch.reshape(fine,[s,150,3])
        #fine[~mask] = 0
        fine = torch.reshape(fine,[s,3,150])
        #print("213",fine.shape)

        if torch.isnan(fine).any():
          print("fine1 è nan")


 
        fine = fine + center
        s = fine.shape[0]
        fine = torch.reshape(fine,[s,150,3])
        #fine[~mask] = 0
        fine = torch.reshape(fine,[s,3,150])
        #print("fine somma", fine)
        #print("214",fine.shape)
        #fine = self.conv3(F.relu(self.conv2(F.relu(self.conv1(feat))))) + center
        #print("fine", fine.shape)
        if torch.isnan(fine).any():
          print("fine2 è nan")
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

    def forward(self, x, gt=None, mask = None, prefix="train", mean_feature=None, alpha=None):
        
        #print("SHAPE MASCHERA",mask_gt.shape)
        if torch.isnan(x).any():
          print("x first input è nan")
        feat = self.encoder(x)
        if torch.isnan(feat).any():
          print("feat after encoder è nan")
        #print("feat", feat.shape)
        out1, out2 = self.decoder(feat, mask)
        out1 = out1.transpose(1, 2).contiguous()
        out2 = out2.transpose(1, 2).contiguous()

        if torch.isnan(out2).any():
          print("out2 after decoder è nan")

        #out1[~mask_gt] = 0
        out2[~mask] = 0
        a = 0.32
        b = 0.36
        g = 0.32
        
        
        
        if prefix=="train":
            if self.train_loss == 'emd':
                loss1 = calc_emd(out1, gt)
                loss2 = calc_emd(out2, gt)
            elif self.train_loss == 'cd':
                
                lossMSE = nn.MSELoss(reduction = 'none')
                loss2MSE = lossMSE(out2,gt)

                #print("loss2MSE", loss2MSE.size())
                #if math.isnan(loss2MSE.mean()):
                #  print("loss2MSE è nan")
                
                # Classic CD
                loss1cd, _ = calc_cd(out1, gt)
                loss2cd, _ = calc_cd(out2, gt)

                #print("loss1cd shape:", loss1cd.size())
                #if math.isnan(loss1cd.mean()):
                #  print("loss1cd è nan")

                #print("loss2cd shape:", loss2cd.size())
                #if math.isnan(loss2cd.mean()):
                #  print("loss2cd è nan")
                
                # DCD
                #loss_opts_alpha = 200
                #loss_opts_lambda = 0.5
                #loss1cd, _, _ = calc_dcd(out1, gt, alpha=loss_opts_alpha, n_lambda=loss_opts_lambda)
                #loss2cd, _, _ = calc_dcd(out2, gt, alpha=loss_opts_alpha, n_lambda=loss_opts_lambda)
                
                #out1pt = out1[:,:,2] # pt
                #out1dis = out1[:,:,-1] # eta,phi

                
                out2_pt = out2[:, :, -1]  # pt
                out2_eta_phi = out2[:, :, :-1]  # eta, phi
                gt_pt = gt[:, :, -1]  # pt
                gt_eta_phi = gt[:, :, :-1]  # eta,phi

                #loss1MSE_pt = lossMSE(out1pt, gtpt)
                loss2MSE_pt = lossMSE(out2_pt, gt_pt)
                #loss1MSE_dis = lossMSE(out1dis, gtdis)
                loss2MSE_dis = lossMSE(out2_eta_phi, gt_eta_phi)


                #print("loss2mse pt ",loss2MSE_pt.size())
                #if math.isnan(loss2MSE_pt.mean()):
                #  print("loss2MSE è nan")
                  
                #print("loss2mse dis ",loss2MSE_dis.size())
                #if math.isnan(loss2MSE_dis.mean()):
                #  print("loss2MSE_dis è nan")

                loss2MSE_totdis = loss2MSE_pt.mean() * loss2MSE_dis.mean()
                #print("loss2MSE_totdis shape", loss2MSE_totdis.shape)
                
                #loss2 = (a * loss2MSE) + (b * loss2cd) + (g * loss2MSE_totdis)
                
            else:
                raise NotImplementedError('Train loss is either CD or EMD!')

            total_train_loss_cd = loss1cd.mean() + loss2cd.mean() * alpha  #ATTENZIONE, ALPHA PRE-ESISTENTE
            #total_train_loss_cd =  loss2cd.mean() * alpha  #ATTENZIONE, ALPHA PRE-ESISTENTE
            total_train_loss_MSE = loss2MSE.mean()
            total_train_loss_MSE_dis = loss2MSE_totdis.mean()
            #print("ttl_MSE_dis shape: ",total_train_loss_MSE_dis.shape)

             
            total_train_loss = (a * total_train_loss_MSE) + (b*total_train_loss_cd) + (g*total_train_loss_MSE_dis)
            #(total_train_loss = loss_cd * alpha)
            
            return out2, loss2cd, total_train_loss
        elif prefix=="val":
            if self.eval_emd:
                emd = calc_emd(out2, gt, eps=0.004, iterations=3000)
            else:
                emd = 0
            cd_p, cd_t, f1 = calc_cd(out2, gt, calc_f1=True)

            #print("cd_p shape:", cd_p.size())
            #if math.isnan(cd_p.mean()):
            #      print("cd_p è nan")

            lossMSE = nn.MSELoss(reduction = 'mean')
            loss2MSE = lossMSE(out2,gt)

            #print("loss2MSE shape:", loss2MSE.size())
            #if math.isnan(loss2MSE.mean()):
            #      print("loss2MSE è nan")
            
            #loss_opts_alpha = 200
            #loss_opts_lambda = 0.5
            #loss2cd, _, _ = calc_dcd(out2, gt, alpha=loss_opts_alpha, n_lambda=loss_opts_lambda)
           
            out2_pt = out2[:, :, -1]  # pt
            out2_eta_phi = out2[:, :, :-1]  # eta, phi
            gt_pt = gt[:, :, -1]  # pt
            gt_eta_phi = gt[:, :, :-1]  # eta,phi

            #loss1MSE_pt = lossMSE(out1pt, gtpt)
            loss2MSE_pt = lossMSE(out2_pt, gt_pt)
            #loss1MSE_dis = lossMSE(out1dis, gtdis)
            loss2MSE_dis = lossMSE(out2_eta_phi, gt_eta_phi)
            loss2MSE_totdis = loss2MSE_dis.mean() * loss2MSE_pt.mean() 


            #print("loss2mse pt shape ",loss2MSE_pt.size())
            #if math.isnan(loss2MSE_pt.mean()):
            #      print("loss2MSE è nan")
                  
            #print("loss2mse dis shape ",loss2MSE_dis.size())
            #if math.isnan(loss2MSE_dis.mean()):
            #      print("loss2MSE_dis è nan")


            total_train_loss_cd = cd_p.mean()
            #total_train_loss_cd = cd_p
            total_train_loss_MSE = loss2MSE.mean()
            total_train_loss_MSE_dis = loss2MSE_totdis.mean()
 

            loss2 = (a * total_train_loss_MSE) + (b * total_train_loss_cd) + (g * total_train_loss_MSE_dis)
            return {'out1': out1, 'out2': out2, 'emd': emd, 'cd_p': cd_p, 'cd_t': cd_t, 'f1': f1, 'tot_loss': loss2}
        else:
            return {'result': out2}
