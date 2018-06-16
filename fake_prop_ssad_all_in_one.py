import torch
import torch.nn as nn
import numpy as np

from dataset import DataSet

device = torch.device('cuda')

def default_box(layer_steps,scale,a_ratios):
    width_set=[scale*ratio for ratio in a_ratios]
    center_set=[1./layer_steps*i+0.5/layer_steps for i in range(layer_steps)]
    width_default=[]
    center_default=[]
    for i in range(layer_steps):
        for j in range(len(a_ratios)):
            width_default.append(width_set[j])
            center_default.append(center_set[i])
    width_default=np.array(width_default)
    center_default=np.array(center_default)
    return width_default,center_default


# IoU Score between anchor and box
def jaccard_with_anchors(anchors_min,anchors_max,len_anchors,box_min,box_max):
    """Compute jaccard score between a box and the anchors.
    """
    anchors_min[anchors_min<box_min] = box_min
    int_xmin = anchors_min

    anchors_max[anchors_max>box_max] = box_max
    int_xmax = anchors_max

    inter_len = int_xmax - int_xmin
    inter_len[inter_len<0] = 0

    union_len = len_anchors - inter_len +box_max-box_min
    jaccard = inter_len / union_len
    return jaccard


class BaseLayers(nn.Module):

    def __init__(self,in_channel,modelkind='B'):

        super(BaseLayers,self).__init__()

        self.modelkind = modelkind
        self.in_channel = in_channel

        if modelkind=='B':
            self.baselayer = nn.Sequential(
                nn.Conv1d(in_channels=self.in_channel,out_channels=256,kernel_size=9,stride=1,padding=4),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=4,stride=4),
                nn.Conv1d(in_channels=256,out_channels=256,kernel_size=9,stride=1,padding=4),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=4,stride=4),
            )
        elif modelkind=='B2':
            self.baselayer = nn.Sequential(
                nn.Conv1d(in_channels=self.in_channel,out_channels=256,kernel_size=9,stride=1,padding=4),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2,stride=2),
                nn.Conv1d(in_channels=256,out_channels=256,kernel_size=9,stride=1,padding=4),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2,stride=2),
            )

    def forward(self, x):
        return self.baselayer(x)

class AnchorLayer(nn.Module):

    def __init__(self,config,layer_name):

        super(AnchorLayer,self).__init__()

        self.config = config
        self.conv1d = nn.Conv1d(512,config.num_dbox[layer_name]*1,kernel_size=1,stride=1)

    def forward(self, x):
        x = self.conv1d(x)
        x = torch.transpose(x,1,2)
        x = x.contiguous()
        n = x.shape[0]
        x = x.view(n,-1,1)
        return x

class SSAD(nn.Module):

    # Prop_SSAD
    def __init__(self,config):

        super(SSAD,self).__init__()

        self.config = config
        self.featuresize = self.config.n_inputs
        self.baselayer = BaseLayers(self.featuresize,modelkind='B2')
        self.relu = nn.ReLU()

        # 64
        self.conv_A1 = nn.Conv1d(256,512,kernel_size=3,stride=2,padding=1)
        self.anchor_layer1 = AnchorLayer(self.config,'conv6')

        # 32
        self.conv_A2 = nn.Conv1d(512,512,kernel_size=3,stride=2,padding=1)
        self.anchor_layer2 = AnchorLayer(self.config,'conv7')

        # 16
        self.conv_A3 = nn.Conv1d(512,512,kernel_size=3,stride=2,padding=1)
        self.anchor_layer3 = AnchorLayer(self.config,'conv8')

        # 8
        self.conv_A4 = nn.Conv1d(512,512,kernel_size=3,stride=2,padding=1)
        self.anchor_layer4 = AnchorLayer(self.config,'conv9')

        # 4
        self.conv_A5 = nn.Conv1d(512,512,kernel_size=3,stride=2,padding=1)
        self.anchor_layer5 = AnchorLayer(self.config,'conv10')

        # 2
        self.conv_A6 = nn.Conv1d(512,512,kernel_size=3,stride=2,padding=1)
        self.anchor_layer6 = AnchorLayer(self.config,'conv11')

        # 1
        self.conv_A7 = nn.Conv1d(512,512,kernel_size=3,stride=2,padding=1)
        self.anchor_layer7 = AnchorLayer(self.config,'conv12')


    def forward(self, x):

        x = self.baselayer(x)
        x = self.conv_A1(x)
        x = self.relu(x)
        anchor_1 = self.anchor_layer1(x)
        x = self.conv_A2(x)
        x = self.relu(x)
        anchor_2 = self.anchor_layer2(x)
        x = self.conv_A3(x)
        x = self.relu(x)
        anchor_3 = self.anchor_layer3(x)
        x = self.conv_A4(x)
        x = self.relu(x)
        anchor_4 = self.anchor_layer4(x)
        x = self.conv_A5(x)
        x = self.relu(x)
        anchor_5 = self.anchor_layer5(x)
        x = self.conv_A6(x)
        x = self.relu(x)
        anchor_6 = self.anchor_layer6(x)
        x = self.conv_A7(x)
        x = self.relu(x)
        anchor_7 = self.anchor_layer7(x)


        return anchor_1,anchor_2,anchor_3,anchor_4,anchor_5,anchor_6,anchor_7

Sigmoid = nn.Sigmoid()

# 把anchor的输出和用对应的default_box来综合处理一下,返回这个anchor预测的类别,置信度,中心点,宽度
def SSAD_box_adjust(anchors,config,layer_name):

    dboxes_w,dboxes_x = default_box(config.num_anchors[layer_name],config.scale[layer_name],config.aspect_ratios[layer_name])

    dboxes_w = torch.from_numpy(dboxes_w).float().to(device)
    dboxes_x = torch.from_numpy(dboxes_x).float().to(device)

    # sigmoid for iou
    anchors_conf = Sigmoid(anchors[:,:,0])

    anchors_rx = torch.ones_like(anchors_conf)
    anchors_rw = torch.ones_like(anchors_conf)

    anchors_rx=anchors_rx*dboxes_w*0.1+dboxes_x
    anchors_rw=torch.exp(0.1*anchors_rw)*dboxes_w

    return anchors_conf,anchors_rw,anchors_rx


def SSAD_bboxes_encode(anchors,glabels,gbboxes,gIndexs,config,layer_name):

    num_anchors = config.num_anchors[layer_name]
    num_dbox = config.num_dbox[layer_name]
    batch_size = config.batch_size

    anchors_conf,anchors_rw,anchors_rx = SSAD_box_adjust(anchors,config,layer_name)

    batch_match_x = []
    batch_match_w = []
    batch_match_scores = []

    # for every video
    for i in range(batch_size):

        b_anchors_rx = anchors_rx[i]
        b_anchors_rw = anchors_rw[i]

        b_glabals = glabels[gIndexs[i]:gIndexs[i+1]]
        b_gbboxes = gbboxes[gIndexs[i]:gIndexs[i+1]]

        n = b_glabals.shape[0]

        shape = num_anchors*num_dbox
        match_x = torch.zeros(shape).to(device)
        match_w = torch.zeros(shape).to(device)
        match_scores = torch.zeros(shape).to(device)

        # 对每一个ground_truth, 都要找anchor上进行配对
        for j in range(n):

            box_min = b_gbboxes[j,0]
            box_max = b_gbboxes[j,1]
            box_x = (box_max+box_min)/2
            box_w = box_max-box_min

            anchors_min = b_anchors_rx - b_anchors_rw / 2
            anchors_max = b_anchors_rx + b_anchors_rw / 2
            len_anchors = anchors_max - anchors_min

            jaccards = jaccard_with_anchors(anchors_min,anchors_max,len_anchors,box_min,box_max)

            mask = jaccards >= match_scores
            # 阀值
            mask = mask & (jaccards > 0.5)

            mask = mask.float()

            match_x = box_x * mask + (1-mask) * match_x
            match_w = box_w * mask + (1-mask) * match_w

            match_scores = torch.max( jaccards , match_scores)

        batch_match_scores.append(match_scores)
        batch_match_x.append(match_x)
        batch_match_w.append(match_w)

    batch_match_x = torch.cat(batch_match_x)
    batch_match_w = torch.cat(batch_match_w)
    batch_match_scores = torch.cat(batch_match_scores)

    return batch_match_scores,batch_match_w,batch_match_x


import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

class SmoothL1Loss(_Loss):
    def __init__(self, size_average=True, reduce=True):
        super(SmoothL1Loss, self).__init__(size_average, reduce)

    def forward(self, input, target):
        target = target.detach()
        return F.smooth_l1_loss(input, target, size_average=self.size_average,
                                reduce=self.reduce)

Smooth_L1 = SmoothL1Loss()

def SSAD_loss(anchors_conf,anchors_xmin,anchors_xmax,match_x,match_w,match_scores,config):

    match_xmin = match_x - match_w/2
    match_xmax = match_x + match_w/2

    num_entries = match_scores.shape[0]

    pmask = match_scores > 0.5
    num_positive = torch.sum(pmask)

    hmask = match_scores < 0.5
    hmask = hmask & (anchors_conf > 0.5)
    num_hard = torch.sum(hmask)

    r_negative=(config.negative_ratio-num_hard.item()/num_positive.item()) * num_positive.item()/(num_entries-num_positive.item()-num_hard.item())
    if r_negative > 1 : r_negative = 1.0

    nmask = torch.FloatTensor(pmask.shape[0]).uniform_().to(device)
    nmask = nmask*(1-pmask.float())
    nmask = nmask*(1-hmask.float())
    nmask[nmask>(1.0-r_negative)] = 1
    nmask[nmask<=(1.0-r_negative)] = 0
    nmask = nmask.byte()

    # loc loss
    mask_loc = pmask
    mask_loc = torch.nonzero(mask_loc).view(-1)
    anchors_xmin = torch.index_select(anchors_xmin,index=mask_loc,dim=0)
    anchors_xmax = torch.index_select(anchors_xmax,index=mask_loc,dim=0)
    match_xmin = torch.index_select(match_xmin,index=mask_loc,dim=0)
    match_xmax = torch.index_select(match_xmax,index=mask_loc,dim=0)
    
    loc_loss = Smooth_L1(input=anchors_xmin,target=match_xmin) + Smooth_L1(input=anchors_xmax,target=match_xmax)

    # conf loss
    mask_conf = pmask+nmask+hmask
    mask_conf = torch.nonzero(mask_conf).view(-1)
    anchors_conf = torch.index_select(anchors_conf,index=mask_conf,dim=0)
    match_scores = torch.index_select(match_scores,index=mask_conf,dim=0)

    conf_loss = Smooth_L1(input=anchors_conf,target=match_scores)

    print('p_num:',num_positive.item(),'h_num:',num_hard.item(),'n_num:',torch.sum(nmask).item())

    return conf_loss,loc_loss


def SSAD_Train(SSAD_Model,X,glabels,gbboxes,gIndexs,config):
    '''
    X: B x feature x 256
    '''
    anchor_1,anchor_2,anchor_3,anchor_4,anchor_5,anchor_6,anchor_7 = SSAD_Model(X)

    layername = ['conv6','conv7','conv8','conv9','conv10','conv11','conv12']
    layers = {'conv6':anchor_1,'conv7':anchor_2,'conv8':anchor_3,'conv9':anchor_4,
              'conv10':anchor_5,'conv11':anchor_6,'conv12':anchor_7}
    full_anchors_conf = []
    full_anchors_min = []
    full_anchors_max = []
    full_match_scores = []
    full_match_w = []
    full_match_x = []

    for layer_name in layername:

        anchors = layers[layer_name]

        batch_match_scores,batch_match_w,batch_match_x = SSAD_bboxes_encode(anchors,glabels,gbboxes,gIndexs,config,layer_name)

        full_match_scores.append(batch_match_scores)
        full_match_x.append(batch_match_x)
        full_match_w.append(batch_match_w)

        anchors_conf,anchors_rw,anchors_rx = SSAD_box_adjust(anchors,config,layer_name)
        anchors_max = anchors_rx+anchors_rw/2
        anchors_min = anchors_rx-anchors_rw/2

        full_anchors_conf.append(anchors_conf.view(-1))
        full_anchors_max.append(anchors_max.view(-1))
        full_anchors_min.append(anchors_min.view(-1))


    full_match_w = torch.cat(full_match_w)
    full_match_x = torch.cat(full_match_x)
    full_match_scores = torch.cat(full_match_scores)

    full_anchors_conf = torch.cat(full_anchors_conf)
    full_anchors_max = torch.cat(full_anchors_max)
    full_anchors_min = torch.cat(full_anchors_min)

    # conf_loss,loc_loss = SSAD_loss(full_anchors_conf,full_anchors_min,full_anchors_max,full_match_x,full_match_w,full_match_scores,config)
    # loss_all = conf_loss+loc_loss

    conf_loss = SSAD_loss(full_anchors_conf,full_anchors_min,full_anchors_max,full_match_x,full_match_w,full_match_scores,config)
    loss_all = conf_loss

    return loss_all,conf_loss,conf_loss


class Config(object):
    """
    define a class to store parameters,
    the input should be feature mat of training and testing
    """
    def __init__(self):
        #common information
        self.training_epochs = 120
        self.batch_size = 256
        self.input_steps= 512
        self.learning_rates=[0.0001]*30+[0.00001]*50+[0.000001]*100
        self.num_classes=201
        self.n_inputs = 2688 + 2048
        self.negative_ratio = 1.01
        self.scale={'conv6':1./64,'conv7':1./32,'conv8':1./16,'conv9':1./8,'conv10':1./4,'conv11':1./2,'conv12':1}
        self.num_anchors={'conv6':64, 'conv7':32,'conv8':16, 'conv9': 8, 'conv10': 4 , 'conv11':2, 'conv12':1}
        self.aspect_ratios={ 'conv6':[1,1.25,1.5,1.75,2],
                             'conv7':[0.5,0.75,1,1.25,1.5,1.75,2],
                             'conv8':[0.5,0.75,1,1.25,1.5,1.75,2],
                             'conv9':[0.5,0.75,1,1.25,1.5,1.75,2],
                             'conv10':[0.5,0.75,1,1.25,1.5,1.75,2],
                             'conv11':[0.5,0.75,1,1.25,1.5,1.75,2],
                             'conv12':[0.5,0.75,1,1.25]}
        self.num_dbox={'conv6':5,'conv7':7,'conv8':7,'conv9':7,'conv10':7,'conv11':7,'conv12':4}

def save_modle(model,path):
    torch.save(model.state_dict(),path)

def change_optim_lr(optim,nlr):
    for param_group in optim.param_groups:
        if param_group['lr'] != nlr:
            print('change optim lr to {}'.format(nlr),flush=True)
            param_group['lr'] = nlr

def train_it():

    SAVE_PATH = '/mnt/md1/Experiments/SSAD_Test9'

    config = Config()
    ssad = SSAD(config).to(device)
    # optim = torch.optim.SGD(ssad.parameters(),lr=0.5,momentum=0.9,weight_decay=0.0001)
    optim = torch.optim.Adam(ssad.parameters(),lr=config.learning_rates[0],weight_decay=0.0001)

    # dataset_train = DataSet('training',True,'HQZ_DPN107_RGB_FULL')
    # dataset_val   = DataSet('validation',False,'HQZ_DPN107_RGB_FULL')

    dataset_train = DataSet('training',False,'MIX_RES200_DPN107')
    dataset_val   = DataSet('validation',False,'MIX_RES200_DPN107')

    TRAIN_ITER =  len(dataset_train.vids)//config.batch_size+1
    VAL_ITER = len(dataset_val.vids)//config.batch_size+1

    for epoch in range(config.training_epochs):

        ssad.train()

        dataset_train.pemutate_vids()
        dataset_val.pemutate_vids()

        for idx in range(TRAIN_ITER):

            gF,gL,gB,gI = dataset_train.nextbatch(config.batch_size)
            gF = np.transpose(gF,(0,2,1))
            gF = torch.from_numpy(gF).to(device).float()
            gL = torch.from_numpy(gL).to(device).long()
            gB = torch.from_numpy(gB).to(device).float()

            ssad.zero_grad()
            train_loss,_,_ = SSAD_Train(ssad,gF,gL,gB,gI,config)
            train_loss.backward()
            optim.step()

            print('Train: {} {}/{} train_loss: {}'.format(epoch,idx,TRAIN_ITER,train_loss.item()),flush=True)

        if epoch%2==0:
            ssad.eval()
            for idx in range(VAL_ITER):
                with torch.no_grad():
                    gF,gL,gB,gI = dataset_val.nextbatch(config.batch_size)
                    gF = np.transpose(gF,(0,2,1))
                    gF = torch.from_numpy(gF).to(device).float()
                    gL = torch.from_numpy(gL).to(device).long()
                    gB = torch.from_numpy(gB).to(device).float()

                    val_loss,_,_ = SSAD_Train(ssad,gF,gL,gB,gI,config)
                    print('Test: {} {}/{} test_loss: {}'.format(epoch,idx,VAL_ITER,val_loss.item()),flush=True)

            # save model
            save_modle(ssad,SAVE_PATH+'/ssad_resnet200_2048_{:03d}.pth'.format(epoch))
            # change learning rate
            change_optim_lr(optim,config.learning_rates[epoch])


if __name__=='__main__':
    train_it()
