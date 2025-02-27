#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from collections import OrderedDict
from typing import Tuple

import numpy as np
import torch
from hymavi.training.data_augmentation.data_augmentation_moreDA import get_moreDA_augmentation
from hymavi.training.loss_functions.deep_supervision import MultipleOutputLoss2
from hymavi.utilities.to_torch import maybe_to_torch, to_cuda
from hymavi.network_architecture.initialization import InitWeights_He
from hymavi.network_architecture.neural_network import SegmentationNetwork
from hymavi.training.data_augmentation.default_data_augmentation import default_2D_augmentation_params, \
    get_patch_size, default_3D_augmentation_params
from hymavi.training.dataloading.dataset_loading import unpack_dataset
from hymavi.training.network_training.Trainer_amos import Trainer_amos
from hymavi.utilities.nd_softmax import softmax_helper
from sklearn.model_selection import KFold
from torch import nn
from torch.cuda.amp import autocast
from hymavi.training.learning_rate.poly_lr import poly_lr
from batchgenerators.utilities.file_and_folder_operations import *
from fvcore.nn import FlopCountAnalysis
from hymavi.network_architecture.amos.hymavi_amos import HymaviNet #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
from hymavi.training.loss_functions.combined_loss import CombinedLoss #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
from hymavi.network_architecture.hymavi_amos.config import config as cf #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


class hymavi_trainer_amos(Trainer_amos):
    """
    Info for Fabian: same as internal nnUNetTrainerV2_2
    """

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.max_num_epochs = 1600
        self.initial_lr = 1e-2
        self.deep_supervision_scales = None
        self.ds_loss_weights = None
        self.pin_memory = True
        self.load_pretrain_weight = False

        self.load_plans_file()

        if len(self.plans['plans_per_stage']) == 2:
            Stage = 1
        else:
            Stage = 0

        self.crop_size = self.plans['plans_per_stage'][Stage]['patch_size']
        self.input_channels = self.plans['num_modalities']
        self.num_classes = self.plans['num_classes'] + 1
        self.conv_op = nn.Conv3d
        print(f"*********************** There are {self.num_classes} classes in this dataset *****************************")
        self.embedding_dim = 96
        self.depths = [2, 2, 2, 2]
        self.num_heads = [3, 6, 12, 24]
        self.embedding_patch_size = [1, 4, 4]
        self.window_size = [[3, 5, 5], [3, 5, 5], [7, 10, 10], [3, 5, 5]]
        self.down_stride = [[1, 4, 4], [1, 8, 8], [2, 16, 16], [4, 32, 32]]
        self.deep_supervision = cf.is_do_ds #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    def initialize(self, training=True, force_load_plans=False):
        """
        - replaced get_default_augmentation with get_moreDA_augmentation
        - enforce to only run this code once
        - loss function wrapper for deep supervision

        :param training:
        :param force_load_plans:
        :return:
        """
        if not self.was_initialized:
            maybe_mkdir_p(self.output_folder)

            if force_load_plans or (self.plans is None):
                self.load_plans_file()

            self.plans['plans_per_stage'][0]['patch_size'] = np.array(cf.input_res[1:])
            self.crop_size = np.array(cf.input_res[1:])

            self.plans['plans_per_stage'][self.stage]['pool_op_kernel_sizes'] = [[1, 4, 4], [2, 2, 2], [2, 2, 2]]
            self.process_plans(self.plans)
            # self.loss =CombinedLoss() #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            self.setup_DA_params()
            if self.deep_supervision:
                ################# Here we wrap the loss for deep supervision ############
                # we need to know the number of outputs of the network
                net_numpool = len(self.net_num_pool_op_kernel_sizes)

                # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
                # this gives higher resolution outputs more weight in the loss
                weights = np.array([1 / (2 ** i) for i in range(net_numpool)])

                # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
                # mask = np.array([True] + [True if i < net_numpool - 1 else False for i in range(1, net_numpool)])
                # weights[~mask] = 0
                weights = weights / weights.sum()
                print(weights)
                self.ds_loss_weights = weights
                # now wrap the loss
                self.loss = MultipleOutputLoss2(self.loss, self.ds_loss_weights)
                # self.loss =CombinedLoss() #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
                ################# END ###################

            self.folder_with_preprocessed_data = join(self.dataset_directory,
                                                      self.plans['data_identifier'] + "_stage%d" % self.stage)
            seeds_train = np.random.random_integers(0, 99999, self.data_aug_params.get('num_threads'))
            seeds_val = np.random.random_integers(0, 99999, max(self.data_aug_params.get('num_threads') // 2, 1))
            if training:
                self.dl_tr, self.dl_val = self.get_basic_generators()
                if self.unpack_data:
                    print("unpacking dataset")
                    unpack_dataset(self.folder_with_preprocessed_data)
                    print("done")
                else:
                    print(
                        "INFO: Not unpacking data! Training may be slow due to that. Pray you are not using 2d or you "
                        "will wait all winter for your model to finish!")

                self.tr_gen, self.val_gen = get_moreDA_augmentation(
                    self.dl_tr, self.dl_val,
                    self.data_aug_params[
                        'patch_size_for_spatialtransform'],
                    self.data_aug_params,
                    deep_supervision_scales=self.deep_supervision_scales if self.deep_supervision else None,
                    pin_memory=self.pin_memory,
                    use_nondetMultiThreadedAugmenter=False,
                    seeds_train=seeds_train,
                    seeds_val=seeds_val
                )
                self.print_to_log_file("TRAINING KEYS:\n %s" % (str(self.dataset_tr.keys())),
                                       also_print_to_console=False)
                self.print_to_log_file("VALIDATION KEYS:\n %s" % (str(self.dataset_val.keys())),
                                       also_print_to_console=False)
            else:
                pass

            self.initialize_network()
            self.initialize_optimizer_and_scheduler()

            assert isinstance(self.network, (SegmentationNetwork, nn.DataParallel))
        else:
            self.print_to_log_file('self.was_initialized is True, not running self.initialize again')
        self.was_initialized = True

    def initialize_network(self):
        """
        - momentum 0.99
        - SGD instead of Adam
        - self.lr_scheduler = None because we do poly_lr
        - deep supervision = True
        - i am sure I forgot something here

        Known issue: forgot to set neg_slope=0 in InitWeights_He; should not make a difference though
        :return:
        """

        self.network = HymaviNet() #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper
        # Print the network parameters & Flops
        n_parameters = sum(p.numel() for p in self.network.parameters() if p.requires_grad)
        input_res = cf.input_res #(1, 14, 160, 160) #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        input = torch.ones(()).new_empty((1, *input_res), dtype=next(self.network.parameters()).dtype,
                                         device=next(self.network.parameters()).device)
        flops = FlopCountAnalysis(self.network, input)
        model_flops = flops.total()
        print(f"Total trainable parameters: {round(n_parameters * 1e-6, 2)} M")
        print(f"MAdds: {round(model_flops * 1e-9, 2)} G")

    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"
        self.optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                         momentum=0.99, nesterov=True)
        self.lr_scheduler = None

    def run_online_evaluation(self, output, target):
        """
        due to deep supervision the return value and the reference are now lists of tensors. We only need the full
        resolution output because this is what we are interested in in the end. The others are ignored
        :param output:
        :param target:
        :return:
        """
        if self.deep_supervision:
            target = target[0]
            output = output[0]
        else:
            target = target
            output = output
        return super().run_online_evaluation(output, target)

    def validate(self, do_mirroring: bool = True, use_sliding_window: bool = True,
                 step_size: float = 0.5, save_softmax: bool = True, use_gaussian: bool = True, overwrite: bool = True,
                 validation_folder_name: str = 'validation_raw', debug: bool = False, all_in_gpu: bool = False,
                 segmentation_export_kwargs: dict = None, run_postprocessing_on_folds: bool = True):
        """
        We need to wrap this because we need to enforce self.network.do_ds = False for prediction
        """
        ds = self.network.do_ds
        self.network.do_ds = False
        ret = super().validate(do_mirroring=do_mirroring, use_sliding_window=use_sliding_window, step_size=step_size,
                               save_softmax=save_softmax, use_gaussian=use_gaussian,
                               overwrite=overwrite, validation_folder_name=validation_folder_name, debug=debug,
                               all_in_gpu=all_in_gpu, segmentation_export_kwargs=segmentation_export_kwargs,
                               run_postprocessing_on_folds=run_postprocessing_on_folds)

        self.network.do_ds = ds
        return ret

    def predict_preprocessed_data_return_seg_and_softmax(self, data: np.ndarray, do_mirroring: bool = True,
                                                         mirror_axes: Tuple[int] = None,
                                                         use_sliding_window: bool = True, step_size: float = 0.5,
                                                         use_gaussian: bool = True, pad_border_mode: str = 'constant',
                                                         pad_kwargs: dict = None, all_in_gpu: bool = False,
                                                         verbose: bool = True, mixed_precision=True) -> Tuple[
        np.ndarray, np.ndarray]:
        """
        We need to wrap this because we need to enforce self.network.do_ds = False for prediction
        """
        ds = self.network.do_ds
        self.network.do_ds = False
        ret = super().predict_preprocessed_data_return_seg_and_softmax(data,
                                                                       do_mirroring=do_mirroring,
                                                                       mirror_axes=mirror_axes,
                                                                       use_sliding_window=use_sliding_window,
                                                                       step_size=step_size, use_gaussian=use_gaussian,
                                                                       pad_border_mode=pad_border_mode,
                                                                       pad_kwargs=pad_kwargs, all_in_gpu=all_in_gpu,
                                                                       verbose=verbose,
                                                                       mixed_precision=mixed_precision)
        self.network.do_ds = ds
        return ret

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False, is_train=False):
        """
        gradient clipping improves training stability

        :param data_generator:
        :param do_backprop:
        :param run_online_evaluation:
        :return:
        """
        data_dict = next(data_generator)
        data = data_dict['data']
        target = data_dict['target']

        data = maybe_to_torch(data)
        target = maybe_to_torch(target)

        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)

        self.optimizer.zero_grad()

        if self.fp16:
            with autocast():
                output = self.network(data)
                del data

                l = self.loss(output, target)

            if do_backprop:
                self.amp_grad_scaler.scale(l).backward()
                self.amp_grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.amp_grad_scaler.step(self.optimizer)
                self.amp_grad_scaler.update()
        else: #--------------------------------------------------------------------- @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            if is_train:
                output, feat = self.network(data, is_train=is_train)
                del data
                # print("----------------------- ", output.shape, target.shape, feat.shape, self.num_classes)
                l = self.loss(output, target.squeeze(1).long(), feat, self.num_classes)
                # print("++++++++++++ ", l)
            else:
                output = self.network(data)

            if do_backprop:
                l.backward()
                # torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)

                self.optimizer.step()

        if run_online_evaluation:
            self.run_online_evaluation(output, target)

        del target

        return l.detach().cpu().numpy()

    def do_split(self):
        """
        The default split is a 5 fold CV on all available training cases. nnU-Net will create a split (it is seeded,
        so always the same) and save it as splits_final.pkl file in the preprocessed data directory.
        Sometimes you may want to create your own split for various reasons. For this you will need to create your own
        splits_final.pkl file. If this file is present, nnU-Net is going to use it and whatever splits are defined in
        it. You can create as many splits in this file as you want. Note that if you define only 4 splits (fold 0-3)
        and then set fold=4 when training (that would be the fifth split), nnU-Net will print a warning and proceed to
        use a random 80:20 data split.
        :return:
        """
        if self.fold == "all":
            # if fold==all then we use all images for training and validation
            tr_keys = val_keys = list(self.dataset.keys())
        else:
            splits_file = join(self.dataset_directory, "splits_final.pkl")

            # if the split file does not exist we need to create it
            if not isfile(splits_file):
                self.print_to_log_file("Creating new 5-fold cross-validation split...")
                splits = []
                all_keys_sorted = np.sort(list(self.dataset.keys()))
                kfold = KFold(n_splits=5, shuffle=True, random_state=12345)
                for i, (train_idx, test_idx) in enumerate(kfold.split(all_keys_sorted)):
                    train_keys = np.array(all_keys_sorted)[train_idx]
                    test_keys = np.array(all_keys_sorted)[test_idx]
                    splits.append(OrderedDict())
                    splits[-1]['train'] = train_keys
                    splits[-1]['val'] = test_keys
                save_pickle(splits, splits_file)

            else:
                self.print_to_log_file("Using splits from existing split file:", splits_file)
                splits = load_pickle(splits_file)
                self.print_to_log_file("The split file contains %d splits." % len(splits))

            self.print_to_log_file("Desired fold for training: %d" % self.fold)
            splits[self.fold]['train'] = np.array(['amos_0016', 'amos_0405', 'amos_0214', 'amos_0159', 'amos_0047',
                                                    'amos_0254', 'amos_0036', 'amos_0406', 'amos_0288', 'amos_0024',
                                                    'amos_0391', 'amos_0336', 'amos_0195', 'amos_0362', 'amos_0248',
                                                    'amos_0098', 'amos_0048', 'amos_0263', 'amos_0110', 'amos_0113',
                                                    'amos_0330', 'amos_0014', 'amos_0102', 'amos_0341', 'amos_0190',
                                                    'amos_0199', 'amos_0273', 'amos_0170', 'amos_0015', 'amos_0317',
                                                    'amos_0042', 'amos_0133', 'amos_0351', 'amos_0071', 'amos_0403',
                                                    'amos_0307', 'amos_0033', 'amos_0049', 'amos_0337', 'amos_0116',
                                                    'amos_0384', 'amos_0348', 'amos_0069', 'amos_0005', 'amos_0299',
                                                    'amos_0138', 'amos_0374', 'amos_0212', 'amos_0010', 'amos_0081',
                                                    'amos_0230', 'amos_0004', 'amos_0396', 'amos_0301', 'amos_0162',
                                                    'amos_0050', 'amos_0237', 'amos_0135', 'amos_0171', 'amos_0224',
                                                    'amos_0075', 'amos_0143', 'amos_0242', 'amos_0129', 'amos_0380',
                                                    'amos_0264', 'amos_0021', 'amos_0402', 'amos_0249', 'amos_0197',
                                                    'amos_0054', 'amos_0296', 'amos_0166', 'amos_0367', 'amos_0231',
                                                    'amos_0023', 'amos_0180', 'amos_0217', 'amos_0011', 'amos_0058',
                                                    'amos_0215', 'amos_0094', 'amos_0172', 'amos_0358', 'amos_0025',
                                                    'amos_0142', 'amos_0226', 'amos_0088', 'amos_0173', 'amos_0332',
                                                    'amos_0175', 'amos_0115', 'amos_0268', 'amos_0066', 'amos_0076',
                                                    'amos_0276', 'amos_0141', 'amos_0072', 'amos_0350', 'amos_0137',
                                                    'amos_0387', 'amos_0235', 'amos_0124', 'amos_0381', 'amos_0321',
                                                    'amos_0097', 'amos_0001', 'amos_0297', 'amos_0104', 'amos_0044',
                                                    'amos_0272', 'amos_0027', 'amos_0395', 'amos_0239', 'amos_0185',
                                                    'amos_0079', 'amos_0279', 'amos_0059', 'amos_0379', 'amos_0282',
                                                    'amos_0084', 'amos_0109', 'amos_0320', 'amos_0158', 'amos_0401',
                                                    'amos_0349', 'amos_0179', 'amos_0103', 'amos_0225', 'amos_0086',
                                                    'amos_0366', 'amos_0274', 'amos_0154', 'amos_0392', 'amos_0281',
                                                    'amos_0160', 'amos_0408', 'amos_0259', 'amos_0035', 'amos_0045',
                                                    'amos_0302', 'amos_0181', 'amos_0361', 'amos_0245', 'amos_0127',
                                                    'amos_0404', 'amos_0353', 'amos_0126', 'amos_0376', 'amos_0294',
                                                    'amos_0017', 'amos_0410', 'amos_0198', 'amos_0077', 'amos_0089',
                                                    'amos_0378', 'amos_0067', 'amos_0052', 'amos_0118', 'amos_0125',
                                                    'amos_0019', 'amos_0057', 'amos_0078', 'amos_0390', 'amos_0030',
                                                    'amos_0060', 'amos_0134', 'amos_0388', 'amos_0092', 'amos_0007',
                                                    'amos_0147', 'amos_0186', 'amos_0083', 'amos_0371', 'amos_0188',
                                                    'amos_0383', 'amos_0119', 'amos_0370', 'amos_0064', 'amos_0398',
                                                    'amos_0161', 'amos_0099', 'amos_0043', 'amos_0105', 'amos_0038',
                                                    'amos_0400', 'amos_0192', 'amos_0111', 'amos_0131', 'amos_0184',
                                                    'amos_0196', 'amos_0121', 'amos_0193', 'amos_0149', 'amos_0152',
                                                    'amos_0177', 'amos_0156', 'amos_0006', 'amos_0009', 'amos_0153'])
                                                   
            splits[self.fold]['val'] = np.array(['amos_0155', 'amos_0140', 'amos_0258', 'amos_0051', 'amos_0377',
                                                'amos_0308', 'amos_0136', 'amos_0150', 'amos_0219', 'amos_0120',
                                                'amos_0070', 'amos_0334', 'amos_0041', 'amos_0112', 'amos_0206',
                                                'amos_0144', 'amos_0373', 'amos_0228', 'amos_0013', 'amos_0128',
                                                'amos_0313', 'amos_0132', 'amos_0368', 'amos_0352', 'amos_0090',
                                                'amos_0363', 'amos_0309', 'amos_0123', 'amos_0385', 'amos_0290'])
            
            #---------------for testing, uncomment below-------------------------------------
            '''splits[self.fold]['val'] = np.array(['amos_0174', 'amos_0397', 'amos_0339', 'amos_0032', 'amos_0189',
                                                'amos_0304', 'amos_0087', 'amos_0372', 'amos_0278', 'amos_0200',
                                                'amos_0191', 'amos_0247', 'amos_0176', 'amos_0365', 'amos_0311',
                                                'amos_0157', 'amos_0022', 'amos_0218', 'amos_0034', 'amos_0008',
                                                'amos_0255', 'amos_0040', 'amos_0399', 'amos_0326', 'amos_0018',
                                                'amos_0364', 'amos_0208', 'amos_0061', 'amos_0056', 'amos_0280',
                                                'amos_0085', 'amos_0117', 'amos_0325', 'amos_0063', 'amos_0409',
                                                'amos_0223', 'amos_0194', 'amos_0106', 'amos_0286', 'amos_0073',
                                                'amos_0233', 'amos_0108', 'amos_0244', 'amos_0167', 'amos_0289',
                                                'amos_0029', 'amos_0283', 'amos_0216', 'amos_0203', 'amos_0356',
                                                'amos_0202', 'amos_0333', 'amos_0250', 'amos_0316', 'amos_0328',
                                                'amos_0293', 'amos_0284', 'amos_0346', 'amos_0310', 'amos_0357',
                                                'amos_0323', 'amos_0344', 'amos_0318', 'amos_0257', 'amos_0238',
                                                'amos_0207', 'amos_0204', 'amos_0287', 'amos_0292', 'amos_0342'])'''

            if self.fold < len(splits):
                tr_keys = splits[self.fold]['train']
                val_keys = splits[self.fold]['val']
                self.print_to_log_file("This split has %d training and %d validation cases."
                                       % (len(tr_keys), len(val_keys)))
            else:
                self.print_to_log_file("INFO: You requested fold %d for training but splits "
                                       "contain only %d folds. I am now creating a "
                                       "random (but seeded) 80:20 split!" % (self.fold, len(splits)))
                # if we request a fold that is not in the split file, create a random 80:20 split
                rnd = np.random.RandomState(seed=12345 + self.fold)
                keys = np.sort(list(self.dataset.keys()))
                idx_tr = rnd.choice(len(keys), int(len(keys) * 0.8), replace=False)
                idx_val = [i for i in range(len(keys)) if i not in idx_tr]
                tr_keys = [keys[i] for i in idx_tr]
                val_keys = [keys[i] for i in idx_val]
                self.print_to_log_file("This random 80:20 split has %d training and %d validation cases."
                                       % (len(tr_keys), len(val_keys)))

        tr_keys.sort()
        val_keys.sort()
        # print("---------------------------------------------------------Available dataset keys:", self.dataset.keys())

        self.dataset_tr = OrderedDict()
        for i in tr_keys:
            self.dataset_tr[i] = self.dataset[i]
        self.dataset_val = OrderedDict()
        for i in val_keys:
            self.dataset_val[i] = self.dataset[i]

    def setup_DA_params(self):
        """
        - we increase roation angle from [-15, 15] to [-30, 30]
        - scale range is now (0.7, 1.4), was (0.85, 1.25)
        - we don't do elastic deformation anymore

        :return:
        """

        self.deep_supervision_scales = [[1, 1, 1]] + list(list(i) for i in 1 / np.cumprod(
            np.vstack(self.net_num_pool_op_kernel_sizes), axis=0))[:-1]

        if self.threeD:
            self.data_aug_params = default_3D_augmentation_params
            self.data_aug_params['rotation_x'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            self.data_aug_params['rotation_y'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            self.data_aug_params['rotation_z'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            if self.do_dummy_2D_aug:
                self.data_aug_params["dummy_2D"] = True
                self.print_to_log_file("Using dummy2d data augmentation")
                self.data_aug_params["elastic_deform_alpha"] = \
                    default_2D_augmentation_params["elastic_deform_alpha"]
                self.data_aug_params["elastic_deform_sigma"] = \
                    default_2D_augmentation_params["elastic_deform_sigma"]
                self.data_aug_params["rotation_x"] = default_2D_augmentation_params["rotation_x"]
        else:
            self.do_dummy_2D_aug = False
            if max(self.patch_size) / min(self.patch_size) > 1.5:
                default_2D_augmentation_params['rotation_x'] = (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi)
            self.data_aug_params = default_2D_augmentation_params
        self.data_aug_params["mask_was_used_for_normalization"] = self.use_mask_for_norm

        if self.do_dummy_2D_aug:
            self.basic_generator_patch_size = get_patch_size(self.patch_size[1:],
                                                             self.data_aug_params['rotation_x'],
                                                             self.data_aug_params['rotation_y'],
                                                             self.data_aug_params['rotation_z'],
                                                             self.data_aug_params['scale_range'])
            self.basic_generator_patch_size = np.array([self.patch_size[0]] + list(self.basic_generator_patch_size))
            patch_size_for_spatialtransform = self.patch_size[1:]
        else:
            self.basic_generator_patch_size = get_patch_size(self.patch_size, self.data_aug_params['rotation_x'],
                                                             self.data_aug_params['rotation_y'],
                                                             self.data_aug_params['rotation_z'],
                                                             self.data_aug_params['scale_range'])
            patch_size_for_spatialtransform = self.patch_size

        self.data_aug_params["scale_range"] = (0.7, 1.4)
        self.data_aug_params["do_elastic"] = False
        self.data_aug_params['selected_seg_channels'] = [0]
        self.data_aug_params['patch_size_for_spatialtransform'] = patch_size_for_spatialtransform

        self.data_aug_params["num_cached_per_thread"] = 2

    def maybe_update_lr(self, epoch=None):
        """
        if epoch is not None we overwrite epoch. Else we use epoch = self.epoch + 1

        (maybe_update_lr is called in on_epoch_end which is called before epoch is incremented.
        herefore we need to do +1 here)

        :param epoch:
        :return:
        """
        if epoch is None:
            ep = self.epoch + 1
        else:
            ep = epoch
        self.optimizer.param_groups[0]['lr'] = poly_lr(ep, self.max_num_epochs, self.initial_lr, 1.8)
        self.print_to_log_file("lr:", np.round(self.optimizer.param_groups[0]['lr'], decimals=6))

    def on_epoch_end(self):
        """
        overwrite patient-based early stopping. Always run to 1000 epochs
        :return:
        """
        super().on_epoch_end()
        continue_training = self.epoch < self.max_num_epochs

        # it can rarely happen that the momentum of nnUNetTrainerV2 is too high for some dataset. If at epoch 100 the
        # estimated validation Dice is still 0 then we reduce the momentum from 0.99 to 0.95
        if self.epoch == 100:
            if self.all_val_eval_metrics[-1] == 0:
                self.optimizer.param_groups[0]["momentum"] = 0.95
                self.network.apply(InitWeights_He(1e-2))
                self.print_to_log_file("At epoch 100, the mean foreground Dice was 0. This can be caused by a too "
                                       "high momentum. High momentum (0.99) is good for datasets where it works, but "
                                       "sometimes causes issues such as this one. Momentum has now been reduced to "
                                       "0.95 and network weights have been reinitialized")
        return continue_training

    def run_training(self):
        """
        if we run with -c then we need to set the correct lr for the first epoch, otherwise it will run the first
        continued epoch with self.initial_lr

        we also need to make sure deep supervision in the network is enabled for training, thus the wrapper
        :return:
        """
        self.maybe_update_lr(self.epoch)  # if we dont overwrite epoch then self.epoch+1 is used which is not what we
        # want at the start of the training
        ds = self.network.do_ds
        if self.deep_supervision:
            self.network.do_ds = True
        else:
            self.network.do_ds = False
        ret = super().run_training()
        self.network.do_ds = ds
        return ret
