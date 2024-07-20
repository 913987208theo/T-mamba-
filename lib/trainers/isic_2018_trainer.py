import os
import time
import numpy as np
import datetime

import nni
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from lib import utils
import torch.distributed as dist


class ISIC2018Trainer:
    """
    Trainer class
    """

    def __init__(self, opt, train_loader, valid_loader, model, optimizer, lr_scheduler, loss_function, metric):

        self.opt = opt
        self.train_data_loader = train_loader
        self.valid_data_loader = valid_loader
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_function = loss_function
        self.metric = metric
        self.device = opt["device"]
        self.local_rank = opt["local_rank"]

        if not self.opt["optimize_params"]:
            if self.opt["resume"] is None:
                self.execute_dir = os.path.join(opt["run_dir"], utils.datestr() + "_" + opt["model_name"] + "_" + opt["dataset_name"])
            else:
                self.execute_dir = os.path.dirname(os.path.dirname(self.opt["resume"]))
            self.checkpoint_dir = os.path.join(self.execute_dir, "checkpoints")
            self.tensorboard_dir = os.path.join(self.execute_dir, "board")
            self.log_txt_path = os.path.join(self.execute_dir, "log.txt")
            if self.opt["resume"] is None:
                if not opt["multi_gpu"]:
                    utils.make_dirs(self.checkpoint_dir)
                    utils.make_dirs(self.tensorboard_dir)
                    utils.pre_write_txt("Complete the initialization of model:{}, optimizer:{}, and lr_scheduler:{}".format(self.opt["model_name"], self.opt["optimizer_name"], self.opt["lr_scheduler_name"]), self.log_txt_path)
                else:
                    if opt['local_rank'] == 0:
                        utils.make_dirs(self.checkpoint_dir)
                        utils.make_dirs(self.tensorboard_dir)
                        utils.pre_write_txt("Complete the initialization of model:{}, optimizer:{}, and lr_scheduler:{}".format(self.opt["model_name"], self.opt["optimizer_name"], self.opt["lr_scheduler_name"]), self.log_txt_path)

        
        self.start_epoch = self.opt["start_epoch"]
        self.end_epoch = self.opt["end_epoch"]
        self.best_metric = opt["best_metric"]
        self.terminal_show_freq = opt["terminal_show_freq"]
        self.save_epoch_freq = opt["save_epoch_freq"]

        self.statistics_dict = self.init_statistics_dict()



##训练方法：
    def training(self):
        for epoch in range(self.start_epoch, self.end_epoch):
            self.reset_statistics_dict()  #函数重置统计数据。

            self.optimizer.zero_grad()  #梯度清零

            if self.opt["multi_gpu"]:   #设置多 GPU 训练的 epoch
                self.train_data_loader.sampler.set_epoch(epoch)
            
            self.train_epoch(epoch)  #训练一个 epoch

            self.valid_epoch(epoch)   #验证一个 epoch
            
            # for waiting all processes  #在多 GPU 环境下，等待所有进程完成当前 epoch 的训练和验证。
            if self.opt["multi_gpu"]:
                torch.distributed.barrier()

            if self.local_rank == 0:   #计算和输出统计数据（仅在主进程）
                train_class_IoU = self.statistics_dict["train"]["total_area_intersect"] / self.statistics_dict["train"]["total_area_union"]
                train_class_IoU = np.nan_to_num(train_class_IoU)
                valid_class_IoU = self.statistics_dict["valid"]["total_area_intersect"] / self.statistics_dict["valid"]["total_area_union"]
                valid_class_IoU = np.nan_to_num(valid_class_IoU)
                valid_dsc = self.statistics_dict["valid"]["DSC_sum"] / self.statistics_dict["valid"]["count"]
                valid_JI = self.statistics_dict["valid"]["JI_sum"] / self.statistics_dict["valid"]["count"]
                valid_ACC = self.statistics_dict["valid"]["ACC_sum"] / self.statistics_dict["valid"]["count"]

                if isinstance(self.lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.lr_scheduler.step(valid_JI)
                else:
                    self.lr_scheduler.step()

                print(
                    "[{}]  epoch:[{:05d}/{:05d}]  lr:{:.6f}  train_loss:{:.6f}  train_DSC:{:.6f}  train_IoU:{:.6f}  train_ACC:{:.6f}  train_JI:{:.6f}  valid_DSC:{:.6f}  valid_IoU:{:.6f}  valid_ACC:{:.6f}  valid_JI:{:.6f}  best_JI:{:.6f}"
                    .format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            epoch, self.end_epoch - 1,
                            self.optimizer.param_groups[0]['lr'],
                            self.statistics_dict["train"]["loss"] / self.statistics_dict["train"]["count"],
                            self.statistics_dict["train"]["DSC_sum"] / self.statistics_dict["train"]["count"],
                            train_class_IoU[1],
                            self.statistics_dict["train"]["ACC_sum"] / self.statistics_dict["train"]["count"],
                            self.statistics_dict["train"]["JI_sum"] / self.statistics_dict["train"]["count"],
                            valid_dsc,
                            valid_class_IoU[1],
                            valid_ACC,
                            valid_JI,
                            self.best_metric))
                if not self.opt["optimize_params"]:
                    utils.pre_write_txt(
                        "[{}]  epoch:[{:05d}/{:05d}]  lr:{:.6f}  train_loss:{:.6f}  train_DSC:{:.6f}  train_IoU:{:.6f}  train_ACC:{:.6f}  train_JI:{:.6f}  valid_DSC:{:.6f}  valid_IoU:{:.6f}  valid_ACC:{:.6f}  valid_JI:{:.6f}  best_JI:{:.6f}"
                        .format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                epoch, self.end_epoch - 1,
                                self.optimizer.param_groups[0]['lr'],
                                self.statistics_dict["train"]["loss"] / self.statistics_dict["train"]["count"],
                                self.statistics_dict["train"]["DSC_sum"] / self.statistics_dict["train"]["count"],
                                train_class_IoU[1],
                                self.statistics_dict["train"]["ACC_sum"] / self.statistics_dict["train"]["count"],
                                self.statistics_dict["train"]["JI_sum"] / self.statistics_dict["train"]["count"],
                                valid_dsc,
                                valid_class_IoU[1],
                                valid_ACC,
                                valid_JI,
                                self.best_metric), self.log_txt_path)

                if self.opt["optimize_params"]:
                    nni.report_intermediate_result(valid_JI)

        if self.local_rank == 0:
            if self.opt["optimize_params"]:
                nni.report_final_result(self.best_metric)

##单个epoch的训练
    def train_epoch(self, epoch):
        
        self.model.train()  ##设置模型为训练模式，启用 dropout 和 batch normalization。

        #Dropout 是一种正则化技术，用于防止神经网络过拟合
        #Batch Normalization（批归一化）也是一种正则化技术，用于加速神经网络的训练并稳定学习过程


        '''遍历训练数据集的每个 batch：
        将每个 batch 的输入张量和目标标签加载到指定设备（GPU 或 CPU）。
        '''
        for batch_idx, (input_tensor, target) in enumerate(self.train_data_loader):
            input_tensor, target = input_tensor.to(self.device), target.to(self.device)



            self.optimizer.zero_grad()  #梯度清零


            output = self.model(input_tensor)  #前向传播!!!!
            #当你调用一个 PyTorch 模型对象时，实际上是调用了其 __call__ 方法，这个方法会自动调用模型类中定义的 forward 方法


            dice_loss = self.loss_function(output, target)  #计算损失


            if self.opt["multi_gpu"]:  #多 GPU 环境下的损失汇总
                dist.all_reduce(dice_loss, op=dist.ReduceOp.SUM)


            dice_loss.backward() #反向传播,计算梯度
            
            self.optimizer.step()  #使用优化器更新模型参数


            #计算和更新指标
            self.calculate_metric_and_update_statistcs(output.cpu().float(), target.cpu().float(), len(target), dice_loss.cpu(), mode="train")


            '''定期输出训练信息：
            每隔 terminal_show_freq 个 batch，输出一次训练状态信息，包括当前 epoch、batch 进度、学习率、损失、DSC、IoU、ACC 和 JI 等指标
            '''
            if (batch_idx + 1) % self.terminal_show_freq == 0:
                train_class_IoU = self.statistics_dict["train"]["total_area_intersect"] / self.statistics_dict["train"]["total_area_union"]
                train_class_IoU = np.nan_to_num(train_class_IoU)
                if self.local_rank == 0:
                    print("[{}]  epoch:[{:05d}/{:05d}]  step:[{:04d}/{:04d}]  lr:{:.6f}  loss:{:.6f}  dsc:{:.6f}  IoU:{:.6f}  ACC:{:.6f}  JI:{:.6f}"
                        .format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                epoch, self.end_epoch - 1,
                                batch_idx + 1, len(self.train_data_loader),
                                self.optimizer.param_groups[0]['lr'],
                                self.statistics_dict["train"]["loss"] / self.statistics_dict["train"]["count"],
                                self.statistics_dict["train"]["DSC_sum"] / self.statistics_dict["train"]["count"],
                                train_class_IoU[1],
                                self.statistics_dict["train"]["ACC_sum"] / self.statistics_dict["train"]["count"],
                                self.statistics_dict["train"]["JI_sum"] / self.statistics_dict["train"]["count"]))
                if not self.opt["optimize_params"]:
                    utils.pre_write_txt("[{}]  epoch:[{:05d}/{:05d}]  step:[{:04d}/{:04d}]  lr:{:.6f}  loss:{:.6f}  dsc:{:.6f}  IoU:{:.6f}  ACC:{:.6f}  JI:{:.6f}"
                                        .format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                                epoch, self.end_epoch - 1,
                                                batch_idx + 1, len(self.train_data_loader),
                                                self.optimizer.param_groups[0]['lr'],
                                                self.statistics_dict["train"]["loss"] / self.statistics_dict["train"]["count"],
                                                self.statistics_dict["train"]["DSC_sum"] / self.statistics_dict["train"]["count"],
                                                train_class_IoU[1],
                                                self.statistics_dict["train"]["ACC_sum"] / self.statistics_dict["train"]["count"],
                                                self.statistics_dict["train"]["JI_sum"] / self.statistics_dict["train"]["count"]),
                                        self.log_txt_path)


###单个epoch的验证
    def valid_epoch(self, epoch):

        self.model.eval()

        
        if self.local_rank == 0:
            with torch.no_grad():
                for input_tensor, target in tqdm(self.valid_data_loader):
                    input_tensor, target = input_tensor.to(self.device), target.to(self.device)
                    if self.opt["multi_gpu"]:
                        output = self.model.module(input_tensor)
                    else:  
                        output = self.model(input_tensor)
                    self.calculate_metric_and_update_statistcs(output.cpu(), target.cpu(), len(target), mode="valid")

                cur_JI = self.statistics_dict["valid"]["JI_sum"] / self.statistics_dict["valid"]["count"]

                if (not self.opt["optimize_params"]) and (epoch + 1) % self.save_epoch_freq == 0 and self.local_rank==0:
                    self.save(epoch, cur_JI, self.best_metric, type="normal")
                if not self.opt["optimize_params"] and self.local_rank==0:
                    self.save(epoch, cur_JI, self.best_metric, type="latest")
                if cur_JI > self.best_metric:
                    self.best_metric = cur_JI
                    if not self.opt["optimize_params"] and self.local_rank==0:
                        self.save(epoch, cur_JI, self.best_metric, type="best")
        # else:
        #     with torch.no_grad():
        #         for input_tensor, target in (self.valid_data_loader):
        #             input_tensor, target = input_tensor.to(self.device), target.to(self.device)

        #             output = self.model(input_tensor)
        #             self.calculate_metric_and_update_statistcs(output.cpu(), target.cpu(), len(target), mode="valid")

        #         cur_JI = self.statistics_dict["valid"]["JI_sum"] / self.statistics_dict["valid"]["count"]

        #         if (not self.opt["optimize_params"]) and (epoch + 1) % self.save_epoch_freq == 0 and self.local_rank==0:
        #             self.save(epoch, cur_JI, self.best_metric, type="normal")
        #         if not self.opt["optimize_params"] and self.local_rank==0:
        #             self.save(epoch, cur_JI, self.best_metric, type="latest")
        #         if cur_JI > self.best_metric:
        #             self.best_metric = cur_JI
        #             if not self.opt["optimize_params"] and self.local_rank==0:
        #                 self.save(epoch, cur_JI, self.best_metric, type="best")


##计算指标和更新统计数据的方法
    def calculate_metric_and_update_statistcs(self, output, target, cur_batch_size, loss=None, mode="train"):
        mask = torch.zeros(self.opt["classes"])
        unique_index = torch.unique(target).int()
        for index in unique_index:
            mask[index] = 1
        self.statistics_dict[mode]["count"] += cur_batch_size
        for i, class_name in self.opt["index_to_class_dict"].items():
            if mask[i] == 1:
                self.statistics_dict[mode]["class_count"][class_name] += cur_batch_size
        if mode == "train":
            self.statistics_dict[mode]["loss"] += loss.item() * cur_batch_size
        for metric_name, metric_func in self.metric.items():
            if metric_name == "IoU":
                area_intersect, area_union, _, _ = metric_func(output, target)
                self.statistics_dict[mode]["total_area_intersect"] += area_intersect.numpy()
                self.statistics_dict[mode]["total_area_union"] += area_union.numpy()
            elif metric_name == "ACC":
                batch_mean_ACC = metric_func(output, target)
                self.statistics_dict[mode]["ACC_sum"] += batch_mean_ACC * cur_batch_size
            elif metric_name == "JI":
                batch_mean_JI = metric_func(output, target)
                self.statistics_dict[mode]["JI_sum"] += batch_mean_JI * cur_batch_size
            elif metric_name == "DSC":
                batch_mean_DSC = metric_func(output, target)
                self.statistics_dict[mode]["DSC_sum"] += batch_mean_DSC * cur_batch_size
            else:
                per_class_metric = metric_func(output, target)
                per_class_metric = per_class_metric * mask
                self.statistics_dict[mode][metric_name]["avg"] += (torch.sum(per_class_metric) / torch.sum(mask)).item() * cur_batch_size
                for j, class_name in self.opt["index_to_class_dict"].items():
                    self.statistics_dict[mode][metric_name][class_name] += per_class_metric[j].item() * cur_batch_size


##初始化和重置统计数据字典的方法
    def init_statistics_dict(self):
        statistics_dict = {
            "train": {
                metric_name: {class_name: 0.0 for _, class_name in self.opt["index_to_class_dict"].items()}
                for metric_name in self.opt["metric_names"]
            },
            "valid": {
                metric_name: {class_name: 0.0 for _, class_name in self.opt["index_to_class_dict"].items()}
                for metric_name in self.opt["metric_names"]
            }
        }
        statistics_dict["train"]["total_area_intersect"] = np.zeros((self.opt["classes"],))
        statistics_dict["train"]["total_area_union"] = np.zeros((self.opt["classes"],))
        statistics_dict["valid"]["total_area_intersect"] = np.zeros((self.opt["classes"],))
        statistics_dict["valid"]["total_area_union"] = np.zeros((self.opt["classes"],))
        statistics_dict["train"]["JI_sum"] = 0.0
        statistics_dict["valid"]["JI_sum"] = 0.0
        statistics_dict["train"]["ACC_sum"] = 0.0
        statistics_dict["valid"]["ACC_sum"] = 0.0
        statistics_dict["train"]["DSC_sum"] = 0.0
        statistics_dict["valid"]["DSC_sum"] = 0.0
        for metric_name in self.opt["metric_names"]:
            statistics_dict["train"][metric_name]["avg"] = 0.0
            statistics_dict["valid"][metric_name]["avg"] = 0.0
        statistics_dict["train"]["loss"] = 0.0
        statistics_dict["train"]["class_count"] = {class_name: 0 for _, class_name in self.opt["index_to_class_dict"].items()}
        statistics_dict["valid"]["class_count"] = {class_name: 0 for _, class_name in self.opt["index_to_class_dict"].items()}
        statistics_dict["train"]["count"] = 0
        statistics_dict["valid"]["count"] = 0

        return statistics_dict



##重置统计数据字典的方法(在每个新的训练 epoch 开始时，清除上一个 epoch 的统计数据。这个函数确保了每个 epoch 的统计数据是独立的，不会受到前一个 epoch 数据的影响)
    def reset_statistics_dict(self):
        for phase in ["train", "valid"]:
            self.statistics_dict[phase]["count"] = 0
            self.statistics_dict[phase]["total_area_intersect"] = np.zeros((self.opt["classes"],))
            self.statistics_dict[phase]["total_area_union"] = np.zeros((self.opt["classes"],))
            self.statistics_dict[phase]["JI_sum"] = 0.0
            self.statistics_dict[phase]["ACC_sum"] = 0.0
            self.statistics_dict[phase]["DSC_sum"] = 0.0
            for _, class_name in self.opt["index_to_class_dict"].items():
                self.statistics_dict[phase]["class_count"][class_name] = 0
            if phase == "train":
                self.statistics_dict[phase]["loss"] = 0.0
            for metric_name in self.opt["metric_names"]:
                self.statistics_dict[phase][metric_name]["avg"] = 0.0
                for _, class_name in self.opt["index_to_class_dict"].items():
                    self.statistics_dict[phase][metric_name][class_name] = 0.0


## 模型保存和加载的方法
    def save(self, epoch, metric, best_metric, type="normal"):
        state = {
            "epoch": epoch,
            "best_metric": best_metric,
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict()
        }
        if type == "normal":
            save_filename = "{:04d}_{}_{:.4f}.state".format(epoch, self.opt["model_name"], metric)
        else:
            save_filename = '{}_{}.state'.format(type, self.opt["model_name"])
        save_path = os.path.join(self.checkpoint_dir, save_filename)
        torch.save(state, save_path)
        if type == "normal":
            save_filename = "{:04d}_{}_{:.4f}.pth".format(epoch, self.opt["model_name"], metric)
        else:
            save_filename = '{}_{}.pth'.format(type, self.opt["model_name"])
        save_path = os.path.join(self.checkpoint_dir, save_filename)
        if self.opt["multi_gpu"]:
            torch.save(self.model.module.state_dict(), save_path)
        else:
            torch.save(self.model.state_dict(), save_path)

    def load(self):
        if self.opt["resume"] is not None:
            if self.opt["pretrain"] is None:
                raise RuntimeError("Training weights must be specified to continue training")

            resume_state_dict = torch.load(self.opt["resume"], map_location=lambda storage, loc: storage.cuda(self.device))
            self.start_epoch = resume_state_dict["epoch"] + 1
            self.best_metric = resume_state_dict["best_metric"]
            self.optimizer.load_state_dict(resume_state_dict["optimizer"])
            self.lr_scheduler.load_state_dict(resume_state_dict["lr_scheduler"])

            pretrain_state_dict = torch.load(self.opt["pretrain"], map_location=lambda storage, loc: storage.cuda(self.device))
            model_state_dict = self.model.state_dict()
            load_count = 0
            for param_name in model_state_dict.keys():
                if (param_name in pretrain_state_dict) and (model_state_dict[param_name].size() == pretrain_state_dict[param_name].size()):
                    model_state_dict[param_name].copy_(pretrain_state_dict[param_name])
                    load_count += 1
            self.model.load_state_dict(model_state_dict, strict=True)
            print("{:.2f}% of model parameters successfully loaded with training weights".format(100 * load_count / len(model_state_dict)))
            if not self.opt["optimize_params"]:
                utils.pre_write_txt("{:.2f}% of model parameters successfully loaded with training weights".format(100 * load_count / len(model_state_dict)), self.log_txt_path)
        else:
            if self.opt["pretrain"] is not None:
                pretrain_state_dict = torch.load(self.opt["pretrain"], map_location=lambda storage, loc: storage.cuda(self.device))
                model_state_dict = self.model.state_dict()
                load_count = 0
                for param_name in model_state_dict.keys():
                    if (param_name in pretrain_state_dict) and (model_state_dict[param_name].size() == pretrain_state_dict[param_name].size()):
                        model_state_dict[param_name].copy_(pretrain_state_dict[param_name])
                        load_count += 1
                self.model.load_state_dict(model_state_dict, strict=True)
                print("{:.2f}% of model parameters successfully loaded with training weights".format(100 * load_count / len(model_state_dict)))
                if not self.opt["optimize_params"]:
                    utils.pre_write_txt("{:.2f}% of model parameters successfully loaded with training weights".format(100 * load_count / len(model_state_dict)), self.log_txt_path)
