import json
from datetime import datetime
import socket
import os
import torch
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision import transforms
from data import Dataset, Sampler
import transporter
import utils

import pytorch_lightning as pl
import argparse
import sys

parser = argparse.ArgumentParser(description='LUS keypoint network pytorch-lightning parallel')
parser.add_argument('--lr', type=float, default=1e-3, help='')
parser.add_argument('--htmaplam', type=float, default=0.1, help='')
parser.add_argument('--max_epochs', type=int, default=50, help='')
parser.add_argument('--sample_rate', type=int, default=4, help='')
parser.add_argument('--batch_size', type=int, default=32, help='')
parser.add_argument('--num_workers', type=int, default=1, help='')
parser.add_argument('--num_gpus', type=int, default=2, help='')
parser.add_argument('--metric', type=str, default='mse', help='')
parser.add_argument('--name', type=str, default='', help='')
parser.add_argument('--data_root', type=str, default='UltrasoundVideoSummarization/', help='')
parser.add_argument('--LUS_num_chan', type=int, default=10, help='')
parser.add_argument('--LUS_num_keypoints', type=int, default=10, help='')
parser.add_argument('--vq_path', type=str, default='VQVAE_unnorm_trained.pth', help='')

args = parser.parse_args()

print(args.name)
sys.stdout = open('stdout_TPR_' + args.name + '.txt', 'w')
sys.stderr = open('stderr_TPR_' + args.name + '.txt', 'w')



def get_config():
    config = utils.ConfigDict({})
    config.batch_size = args.batch_size
    config.image_channels = args.LUS_num_chan
    config.k = args.LUS_num_keypoints
    config.num_iterations = args.max_epochs * args.batch_size * args.num_gpus
    #Note: originally trained for 1e6 iterations
    config.learning_rate = args.lr
    config.learning_rate_decay_rate = 0.95
    config.learning_rate_decay_every_n_steps = int(args.max_epochs//10)
    return config


def _get_model(config):
    feature_encoder = transporter.FeatureEncoder(config.image_channels)
    pose_regressor = transporter.PoseRegressor(config.image_channels, config.k)
    refine_net = transporter.RefineNet(config.image_channels)

    return transporter.Transporter(feature_encoder, pose_regressor, refine_net)

def _get_data_loader(config):
    transform = transforms.ToTensor()
    dataset = Dataset(config.dataset_root, transform=transform)
    sampler = Sampler(dataset)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=config.batch_size, sampler=sampler, pin_memory=True, num_workers=4)
    return loader

class VQVAEPerceptualLoss(torch.nn.Module):
    def __init__(self, vqvae_path = 'VQVAE_unnorm_trained.pth'):
        super(VQVAEPerceptualLoss, self).__init__()
        encoder = torch.load(vqvae_path)._encoder
        encoder.eval()
        blocks = []

        encoder._residual_stack._layers[0]._block[0].inplace = False
        encoder._residual_stack._layers[0]._block[2].inplace = False
        encoder._residual_stack._layers[1]._block[0].inplace = False
        encoder._residual_stack._layers[1]._block[2].inplace = False
        for module_name, module in encoder.named_modules():
          if module_name == '':
            continue
          if 'residual_stack' in module_name and 'block.' not in module_name:
            continue
          blocks.append(module)

        #for bl in blocks:
        #  bl.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)

    def forward(self, input, target): #(batch_size, 10, 256, 256)
        x = input
        x_ = target
        loss = 0.0
        x = x.view(-1,256,256).unsqueeze(1)
        x_ = x_.view(-1,256,256).unsqueeze(1)
        for block in self.blocks:
            x = block(x)
            x_ = block(x_)
            loss = loss + F.mse_loss(x, x_)
        return loss



class plTransporter(pl.LightningModule):
  def __init__(self, config)
    super().__init__()
    self.model = _get_model(config)
    self.model.train()
    if args.metric == 'mse':
        self.metric = torch.nn.MSELoss()
    elif args.metric == 'perc':
        self.metric = VQVAEPerceptualLoss(args.vq_path)
        
  def forward(self, x1, x2):       
    return self.model(x1, x2)

  def training_step(self, batch, batch_idx):

    x1, x2 = batch
    recovered_x2 = self.forward(x1, x2)
    loss = self.metric(recovered_x2, x2)
    self.log('train_loss',loss)

    return {'loss': loss}

  def validation_step(self, batch, batch_idx):
    x1, x2 = batch
    recovered_x2 = self.forward(x1, x2)
    loss = self.metric(recovered_x2, x2)
    self.monitor = loss
    self.log('val_loss',loss)

    return {'val_loss': loss}

  def training_epoch_end(self, outputs):
    avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
    self.log('train_loss_epoch', avg_loss)

  def validation_epoch_end(self, outputs):
    avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
    self.monitor = avg_loss
    self.log('monitor', self.monitor)
    self.log('val_loss_epoch', avg_loss)


  def configure_optimizers(self):
    self.optimizer = torch.optim.Adam(model.parameters(), config.learning_rate)
    self.scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        config.learning_rate_decay_every_n_steps,
        gamma=config.learning_rate_decay_rate)
    return {"optimizer": self.optimizer, "lr_scheduler": self.scheduler, "monitor": "train_loss"}
  


def main():
    config = get_config()
    print('Running with config\n{}'.format(config))
    key_model = plTransporter(config)

    ROOT = args.data_root
    US_train = USDataset(ROOT + "train/", train=True, sample_rate = args.sample_rate)
    US_test_val = USDataset(ROOT + "val/", train=False, sample_rate = args.sample_rate)
    
    dataset = {}
    dataset["train"], dataset["val"], dataset["test"] = US_train, US_test_val, US_test_val

    train_dataloader = DataLoader(dataset["train"], batch_size=hparams["batch_size"], pin_memory = True, num_workers = args.num_workers)
    val_dataloader = DataLoader(dataset["val"], batch_size=hparams["batch_size"], pin_memory =True,num_workers = args.num_workers )
    
    ckpt_cb = pl.callbacks.ModelCheckpoint('checkpoints/{epoch}-{val_loss:.5f}', monitor='val_loss', verbose=0, save_top_k=6, save_last = True, save_weights_only=False, mode='auto', period=2, prefix='ckpt_AFRESH_' + args.name + '_')
    print("Beginning Trainer!!",sys.stdout)
    trainer = pl.Trainer(#resume_from_checkpoint='checkpoints/ckpt_AFRESH_lr0.00001_-last.ckpt',
                         #gradient_clip_val=1.0,
                         gpus=2, num_nodes=1,accelerator='ddp', max_epochs = args.max_epochs, checkpoint_callback= ckpt_cb)
    trainer.fit(key_model, train_dataloader, val_dataloader)


if __name__ == "__main__":
    main()
