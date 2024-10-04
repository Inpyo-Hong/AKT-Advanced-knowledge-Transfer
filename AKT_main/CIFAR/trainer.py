"""
basic trainer
"""

import time

import torch.autograd
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import utils as utils
import numpy as np
import torch
from pytorchcv.models.resnet import ResUnit
from pytorchcv.models.common import DwsConvBlock
from pytorchcv.models.mobilenetv2 import LinearBottleneck

import math

from sklearn.preprocessing import MinMaxScaler
import numpy as np


__all__ = ["Trainer"]


def compute_entropy(p_logit, T):
    p = F.softmax(p_logit / T, dim=-1)
    entropy = torch.sum(p * (-F.log_softmax(p_logit / T, dim=-1)), 1)

    return entropy


def marginal_loss(teacher_out, student_out, num_class, lambda_upper, lambda_low):
    max_entropy = -(1 / num_class) * math.log(1 / num_class) * num_class
    h_info = compute_entropy(teacher_out - student_out, 1)
    h_info_prime = (h_info - h_info.min()) / (max_entropy - h_info.min())
    zero_vector = torch.zeros(200).cuda()  # batch size 16
    h_loss = torch.mean(torch.max(lambda_low - h_info_prime, zero_vector)) + torch.mean(
        torch.max(h_info_prime - lambda_upper, zero_vector))

    return h_loss


class Trainer(object):
    """
	trainer for training network, use SGD
	"""

    def __init__(self, model, model_teacher, generator, lr_master_S, lr_master_G,
                 train_loader, test_loader, settings, logger, tensorboard_logger=None,
                 opt_type="SGD", optimizer_state=None, run_count=0):
        """
		init trainer
		"""

        self.settings = settings

        self.model = utils.data_parallel(
            model, self.settings.nGPU, self.settings.GPU)
        self.model_teacher = utils.data_parallel(
            model_teacher, self.settings.nGPU, self.settings.GPU)

        self.generator = utils.data_parallel(
            generator, self.settings.nGPU, self.settings.GPU)

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.tensorboard_logger = tensorboard_logger
        self.log_soft = nn.LogSoftmax(dim=1)
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.bce_logits = nn.BCEWithLogitsLoss().cuda()
        self.MSE_loss = nn.MSELoss().cuda()
        self.lr_master_S = lr_master_S
        self.lr_master_G = lr_master_G
        self.opt_type = opt_type
        if opt_type == "SGD":
            self.optimizer_AKT = torch.optim.SGD(
                params=self.model.parameters(),
                lr=self.lr_master_S.lr,
                momentum=self.settings.momentum,
                weight_decay=self.settings.weightDecay,
                nesterov=True,
            )
        elif opt_type == "RMSProp":
            self.optimizer_AKT = torch.optim.RMSprop(
                params=self.model.parameters(),
                lr=self.lr_master_S.lr,
                eps=1.0,
                weight_decay=self.settings.weightDecay,
                momentum=self.settings.momentum,
                alpha=self.settings.momentum
            )
        elif opt_type == "Adam":
            self.optimizer_AKT = torch.optim.Adam(
                params=self.model.parameters(),
                lr=self.lr_master_S.lr,
                eps=1e-5,
                weight_decay=self.settings.weightDecay
            )
        else:
            assert False, "invalid type: %d" % opt_type
        if optimizer_state is not None:
            self.optimizer_AKT.load_state_dict(optimizer_state)

        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.settings.lr_G,
											betas=(self.settings.b1, self.settings.b2))

        self.logger = logger
        self.run_count = run_count
        self.scalar_info = {}
        self.mean_list = []
        self.var_list = []
        self.teacher_running_mean = []
        self.teacher_running_var = []
        self.save_BN_mean = []
        self.save_BN_var = []
        self.activation_teacher = []
        self.activation = []

        self.fix_G = False

    def update_lr(self, epoch):
        """
		update learning rate of optimizers
		:param epoch: current training epoch
		"""
        lr_AKT = self.lr_master_S.get_lr(epoch)
        lr_G = self.lr_master_G.get_lr(epoch)
        # update learning rate of model optimizer
        for param_group in self.optimizer_AKT.param_groups:
            param_group['lr'] = lr_AKT

        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr_G

    def loss_fn_kd(self, output, labels, teacher_outputs, linear=None):
        """
		Compute the knowledge-distillation (KD) loss given outputs, labels.
		"Hyperparameters": temperature and alpha
		"""
        q_loss = 20 * 20 * compute_entropy(teacher_outputs - output, 20)
        q_loss = - torch.mean(q_loss)

        return q_loss

    def spatial_attention(self, x):
        x = F.normalize(x.pow(2).mean([1]).view(x.size(0), -1))
        return x

    def channel_attention(self, x):
        x = x.pow(2).mean([2, 3])
        x = F.softmax(x, dim=1)
        return x


    def compute_kl_div_loss(self, student_attention, teacher_attention):
        """
        Computes the KL divergence loss between student and teacher attention maps.
        Args:
            student_attention (Tensor): The attention values from the student model.
            teacher_attention (Tensor): The attention values from the teacher model.

        Returns:
            kl_div (Tensor): The calculated KL divergence loss.
        """

        FD_T = self.settings.fd_temperature  # Temperature for softening probabilities
        student_probs = F.softmax(student_attention / FD_T, dim=1)
        teacher_probs = F.softmax(teacher_attention / FD_T, dim=1)
        kl_div = F.kl_div(student_probs.log(), teacher_probs, reduction='sum')
        return kl_div

    def loss_refined_fd(self):
       """
       Computes the refined feature distillation (RFD) loss by calculating the
       KL divergence for both channel and spatial attentions between student and teacher models.

       Returns:
           fd_loss (Tensor): The refined feature distillation loss.
       """
        fd = torch.zeros(1).cuda()
        for student_attention, teacher_attention in zip(self.activation, self.activation_teacher):
            channel_student = self.channel_attention(student_attention)
            channel_teacher = self.channel_attention(teacher_attention)
            channel_kl_div = self.compute_kl_div_loss(channel_student, channel_teacher)

            spatial_student = self.spatial_attention(student_attention)
            spatial_teacher = self.spatial_attention(teacher_attention)
            spatial_kl_div = self.compute_kl_div_loss(spatial_student, spatial_teacher)

            fd += channel_kl_div + spatial_kl_div
        fd_loss = self.settings.lam * fd
        return fd_loss

    def hook_activation_teacher(self, module, input, output):
        self.activation_teacher.append(output)


    def hook_activation(self, module, input, output):
        self.activation.append(output)


    def forward(self, images, teacher_outputs, labels=None, linear=None):
        """
		forward propagation
		"""
        self.activation_teacher.clear()
        self.activation.clear()

        output, output_1 = self.model(images, True)
        if labels is not None:
            loss = self.loss_fn_kd(output, labels, teacher_outputs, linear) # logit distillation loss
            loss_RFD = self.loss_refined_fd() #refined feature distillation loss
            loss_AKT = loss + loss_RFD
            return output, loss_AKT
        else:
            return output, None

    def backward_G(self, loss_G):
        """
		backward propagation
		"""
        self.optimizer_G.zero_grad()
        loss_G.backward()
        self.optimizer_G.step()

    def backward_AKT(self, loss_AKT):
        """
		backward propagation
		"""
        self.optimizer_AKT.zero_grad()
        loss_AKT.backward()
        self.optimizer_AKT.step()

    def backward(self, loss):
        """
		backward propagation
		"""
        self.optimizer_G.zero_grad()
        self.optimizer_AKT.zero_grad()
        loss.backward()
        self.optimizer_G.step()
        self.optimizer_AKT.step()

    def hook_fn_forward(self, module, input, output):
        input = input[0]
        mean = input.mean([0, 2, 3])
        # use biased var in train
        var = input.var([0, 2, 3], unbiased=False)

        self.mean_list.append(mean)
        self.var_list.append(var)
        self.teacher_running_mean.append(module.running_mean)
        self.teacher_running_var.append(module.running_var)

    def hook_fn_forward_saveBN(self, module, input, output):
        self.save_BN_mean.append(module.running_mean.cpu())
        self.save_BN_var.append(module.running_var.cpu())


    def train(self, epoch):
        """
		training
		"""
        top1_error = utils.AverageMeter()
        top1_loss = utils.AverageMeter()
        top5_error = utils.AverageMeter()
        fp_acc = utils.AverageMeter()

        iters = 200
        self.update_lr(epoch)

        self.model.eval()
        self.model_teacher.eval()
        self.generator.train()

        start_time = time.time()
        end_time = start_time

        if epoch == 0:
            for m in self.model_teacher.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.register_forward_hook(self.hook_fn_forward)
                if isinstance(m, ResUnit):
                    m.body.register_forward_hook(self.hook_activation_teacher)
                elif isinstance(m, DwsConvBlock):
                    m.pw_conv.bn.register_forward_hook(self.hook_activation_teacher)
                elif isinstance(m, LinearBottleneck):
                    m.conv3.register_forward_hook(self.hook_activation_teacher)
            for m in self.model.modules():
                if isinstance(m, ResUnit):
                    m.body.register_forward_hook(self.hook_activation)
                elif isinstance(m, DwsConvBlock):
                    m.pw_conv.bn.register_forward_hook(self.hook_activation)
                elif isinstance(m, LinearBottleneck):
                    m.conv3.register_forward_hook(self.hook_activation)


        for i in range(iters):
            start_time = time.time()
            data_time = start_time - end_time

            z = Variable(torch.randn(self.settings.batchSize, self.settings.latent_dim)).cuda()

            # Get labels ranging from 0 to n_classes for n rows
            labels = Variable(torch.randint(0, self.settings.nClasses, (self.settings.batchSize,))).cuda()
            z = z.contiguous()
            labels = labels.contiguous()
            images = self.generator(z, labels)

            labels_loss = Variable(torch.zeros(self.settings.batchSize, self.settings.nClasses)).cuda()
            labels_loss.scatter_(1, labels.unsqueeze(1), 1.0)

            self.mean_list.clear()
            self.var_list.clear()
            output_teacher_batch, output_teacher_1 = self.model_teacher(images, out_feature=True)

            output, output_1 = self.model(images, True)

            # generation loss
            z_ds = output_teacher_batch - output
            z_as = output_teacher_batch + output
            loss_ds = ((-(labels_loss * self.log_soft(z_ds)).sum(dim=1))).mean()
            loss_as = ((-(labels_loss * self.log_soft(z_as)).sum(dim=1))).mean()
            loss_onehot = self.settings.alpha_ds * loss_ds + self.settings.alpha_as * loss_as
            h_loss = marginal_loss(output_teacher_batch, output, self.settings.nClasses, self.settings.lambda_u,
                                   self.settings.lambda_l)

            # BN statistic loss
            BNS_loss = torch.zeros(1).cuda()
            for num in range(len(self.mean_list)):
                BNS_loss += self.MSE_loss(self.mean_list[num], self.teacher_running_mean[num]) + self.MSE_loss(
                    self.var_list[num], self.teacher_running_var[num])
            BNS_loss = BNS_loss / len(self.mean_list)

            # loss of Generator
            loss_G = h_loss + self.settings.beta * loss_onehot + self.settings.gamma * BNS_loss
            self.backward_G(loss_G)

            output, loss_AKT = self.forward(images.detach(), output_teacher_batch.detach(), labels, linear=labels_loss)
            if epoch >= self.settings.warmup_epochs:
                self.backward_S(loss_AKT)

            single_error, single_loss, single5_error = utils.compute_singlecrop(
                outputs=output, labels=labels,
                loss=loss_AKT, top5_flag=True, mean_flag=True)

            top1_error.update(single_error, images.size(0))
            top1_loss.update(single_loss, images.size(0))
            top5_error.update(single5_error, images.size(0))

            end_time = time.time()

            gt = labels.data.cpu().numpy()
            d_acc = np.mean(np.argmax(output_teacher_batch.data.cpu().numpy(), axis=1) == gt)

            fp_acc.update(d_acc)




        print(
            "[Epoch %d/%d] [Batch %d/%d] [acc: %.4f%%] [G loss: %f] [Balance loss: %f] [BNS_loss:%f] [AKT loss: %f] "
            % (epoch + 1, self.settings.nEpochs, i + 1, iters, 100 * fp_acc.avg, loss_G.item(),
               loss_onehot.item(), BNS_loss.item(),
               loss_AKT.item())
        )

        self.scalar_info['accuracy every epoch'] = 100 * d_acc
        self.scalar_info['G loss every epoch'] = loss_G
        self.scalar_info['One-hot loss every epoch'] = loss_onehot
        self.scalar_info['AKT loss every epoch'] = loss_AKT

        self.scalar_info['training_top1error'] = top1_error.avg
        self.scalar_info['training_top5error'] = top5_error.avg
        self.scalar_info['training_loss'] = top1_loss.avg

        if self.tensorboard_logger is not None:
            for tag, value in list(self.scalar_info.items()):
                self.tensorboard_logger.scalar_summary(tag, value, self.run_count)
            self.scalar_info = {}

        return top1_error.avg, top1_loss.avg, top5_error.avg

    def test(self, epoch):
        """
		testing
		"""
        top1_error = utils.AverageMeter()
        top1_loss = utils.AverageMeter()
        top5_error = utils.AverageMeter()

        self.model.eval()
        self.model_teacher.eval()

        iters = len(self.test_loader)
        start_time = time.time()
        end_time = start_time

        with torch.no_grad():
            for i, (images, labels) in enumerate(self.test_loader):
                start_time = time.time()

                labels = labels.cuda()
                images = images.cuda()
                output = self.model(images)

                loss = torch.ones(1)
                self.mean_list.clear()
                self.var_list.clear()

                single_error, single_loss, single5_error = utils.compute_singlecrop(
                    outputs=output, loss=loss,
                    labels=labels, top5_flag=True, mean_flag=True)

                top1_error.update(single_error, images.size(0))
                top1_loss.update(single_loss, images.size(0))
                top5_error.update(single5_error, images.size(0))

                end_time = time.time()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [acc: %.4f%%]"
            % (epoch + 1, self.settings.nEpochs, i + 1, iters, (100.00 - top1_error.avg))
        )

        self.scalar_info['testing_top1error'] = top1_error.avg
        self.scalar_info['testing_top5error'] = top5_error.avg
        self.scalar_info['testing_loss'] = top1_loss.avg
        if self.tensorboard_logger is not None:
            for tag, value in self.scalar_info.items():
                self.tensorboard_logger.scalar_summary(tag, value, self.run_count)
            self.scalar_info = {}
        self.run_count += 1

        return top1_error.avg, top1_loss.avg, top5_error.avg

    def test_teacher(self, epoch):
        """
		testing
		"""
        top1_error = utils.AverageMeter()
        top1_loss = utils.AverageMeter()
        top5_error = utils.AverageMeter()

        self.model_teacher.eval()

        iters = len(self.test_loader)
        start_time = time.time()
        end_time = start_time

        with torch.no_grad():
            for i, (images, labels) in enumerate(self.test_loader):
                start_time = time.time()
                data_time = start_time - end_time

                labels = labels.cuda()
                if self.settings.tenCrop:
                    image_size = images.size()
                    images = images.view(
                        image_size[0] * 10, image_size[1] / 10, image_size[2], image_size[3])
                    images_tuple = images.split(image_size[0])
                    output = None
                    for img in images_tuple:
                        if self.settings.nGPU == 1:
                            img = img.cuda()
                        img_var = Variable(img, volatile=True)
                        temp_output, _ = self.forward(img_var)
                        if output is None:
                            output = temp_output.data
                        else:
                            output = torch.cat((output, temp_output.data))
                    single_error, single_loss, single5_error = utils.compute_tencrop(
                        outputs=output, labels=labels)
                else:
                    if self.settings.nGPU == 1:
                        images = images.cuda()

                    output = self.model_teacher(images)

                    loss = torch.ones(1)
                    self.mean_list.clear()
                    self.var_list.clear()

                    single_error, single_loss, single5_error = utils.compute_singlecrop(
                        outputs=output, loss=loss,
                        labels=labels, top5_flag=True, mean_flag=True)
                #
                top1_error.update(single_error, images.size(0))
                top1_loss.update(single_loss, images.size(0))
                top5_error.update(single5_error, images.size(0))

                end_time = time.time()
                iter_time = end_time - start_time

        print(
            "Teacher network: [Epoch %d/%d] [Batch %d/%d] [acc: %.4f%%]"
            % (epoch + 1, self.settings.nEpochs, i + 1, iters, (100.00 - top1_error.avg))
        )

        self.run_count += 1

        return top1_error.avg, top1_loss.avg, top5_error.avg
