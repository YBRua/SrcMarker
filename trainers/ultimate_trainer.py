import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import logging

from models import (TransformSelector, FeatureApproximator)

from tqdm import tqdm
from typing import Dict
from torch.utils.data import DataLoader
from eval_utils import compute_msg_acc

from logger_setup import DefaultLogger
from runtime_data_manager import InMemoryJitRuntimeDataManager
from typing import Optional

from collections import defaultdict


class UltimateWMTrainer:
    def __init__(self,
                 code_encoder: nn.Module,
                 extract_encoder: nn.Module,
                 wm_encoder: nn.Module,
                 selector: TransformSelector,
                 approximator: FeatureApproximator,
                 wm_decoder: nn.Module,
                 scheduled_optimizer: optim.Optimizer,
                 other_optimizer: optim.Optimizer,
                 device: torch.device,
                 train_loader: DataLoader,
                 valid_loader: DataLoader,
                 test_loader: DataLoader,
                 loss_fn: nn.Module,
                 transform_manager: InMemoryJitRuntimeDataManager,
                 w_var: float,
                 w_style: float,
                 scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
                 logger: Optional[logging.Logger] = None,
                 ckpt_dir: str = 'my_model'):

        self.code_encoder = code_encoder
        self.extract_encoder = extract_encoder
        self.selector = selector
        self.approximator = approximator
        self.wm_encoder = wm_encoder
        self.wm_decoder = wm_decoder

        self.main_optimizer = scheduled_optimizer
        self.secondary_optimizer = other_optimizer

        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

        self.transform_manager = transform_manager

        self.warper_loss_fn = nn.MSELoss()
        self.loss_fn = loss_fn

        self.w_var = w_var
        self.w_style = w_style
        self.device = device
        self.logger = logger if logger is not None else DefaultLogger()
        self.scheduler = scheduler

        self.save_dir = os.path.join('./ckpts', ckpt_dir)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.best_metric = 0
        self.var_random_mask = True

    def set_var_random_mask_enabled(self, enable: bool):
        self.var_random_mask = enable

    def _train_epoch(self, eid: int, dataloader: DataLoader):
        var_selection = defaultdict(list)
        self.code_encoder.train()
        if self.extract_encoder is not None:
            self.extract_encoder.train()

        self.selector.train()
        self.approximator.train()
        self.wm_encoder.train()
        self.wm_decoder.train()

        progress = tqdm(dataloader)
        tot_loss = 0.0
        tot_acc = 0.0
        tot_decode_loss = 0.0
        tot_warp_loss = 0.0
        tot_real_acc = 0.0

        for bid, batch in enumerate(progress):
            (x, length, padding_mask, instance_ids, wms, wmids) = batch

            x = x.to(self.device)
            wms = wms.float().to(self.device)
            padding_mask = padding_mask.to(self.device)
            wmids = wmids.to(self.device)

            # get style masks
            s_feasible = self.transform_manager.get_feasible_transform_ids(instance_ids)
            N_TRANS = self.transform_manager.get_transform_capacity()

            s_feasible_01 = []

            for s_f in s_feasible:
                val = torch.ones(N_TRANS, device=self.device).bool()
                val[s_f] = False
                s_feasible_01.append(val)
            s_masks = torch.stack(s_feasible_01, dim=0)

            self.main_optimizer.zero_grad()
            self.secondary_optimizer.zero_grad()

            # original code feature
            code_feature = self.code_encoder(x, length, padding_mask)
            wm_feature = self.wm_encoder(wms)

            # feature warping
            # vs for variable selection, ss for style selection
            vs_output = self.selector.var_selector_forward(
                code_feature, wm_feature, random_mask=self.var_random_mask)
            vs_logits = torch.log_softmax(vs_output, dim=1)
            vs_onehots = F.gumbel_softmax(vs_logits, tau=0.5, hard=True)
            vs_ids = torch.argmax(vs_onehots, dim=1)

            ss_output = self.selector.transform_selector_forward(code_feature,
                                                                 wm_feature,
                                                                 transform_mask=s_masks)
            ss_logits = torch.log_softmax(ss_output, dim=1)
            ss_onehots = F.gumbel_softmax(ss_logits, tau=0.5, hard=True)
            ss_ids = torch.argmax(ss_onehots, dim=1)

            vs_embds = torch.matmul(vs_onehots, self.code_encoder.embedding.weight)
            ss_embds = self.approximator.get_transform_embedding(ss_onehots)
            feat_warped = self.approximator(code_feature, vs_embds, ss_embds,
                                            padding_mask)

            # transformed code feature
            ss_instances = self.transform_manager.get_transformed_codes_by_pred(
                instance_ids, ss_ids.tolist())

            t_instances, _ = self.transform_manager.varname_transform_on_instances(
                ss_instances, vs_ids.tolist())

            for wmid, vs in zip(wmids.tolist(), vs_ids.tolist()):
                word = self.transform_manager.vocab.get_token_by_id(vs)
                var_selection[wmid].append(word)

            tx, tl, tmask = self.transform_manager.load_to_tensor(t_instances)
            tx = tx.to(self.device)
            tmask = tmask.to(self.device)
            if self.extract_encoder is not None:
                tfeatures = self.extract_encoder(tx, tl, tmask)
            else:
                # shared code encoder
                tfeatures = self.code_encoder(tx, tl, tmask)

            # computing feature distance loss
            warper_dist_loss = self.warper_loss_fn(feat_warped, tfeatures)
            feat_warped = feat_warped + torch.randn_like(feat_warped) * 0.1

            # compute decoding loss
            outputs = self.wm_decoder(feat_warped)
            probs = torch.sigmoid(outputs)
            decode_loss = self.loss_fn(probs, wms)
            preds = torch.where(probs > 0.5, 1, 0)

            toutputs = self.wm_decoder(tfeatures)
            tprobs = torch.sigmoid(toutputs)
            t_loss = self.loss_fn(tprobs, wms)

            warper_dist_loss = warper_dist_loss * 0.1
            loss = decode_loss + t_loss + warper_dist_loss
            loss.backward()

            self.main_optimizer.step()
            self.secondary_optimizer.step()

            # simulated decoding
            tpreds = torch.where(tprobs > 0.5, 1, 0)

            # update metrics
            tot_acc += (preds == wms).float().mean().item()
            tot_real_acc += (tpreds == wms).float().mean().item()
            tot_loss += loss.item()
            tot_warp_loss += warper_dist_loss.item()
            tot_decode_loss += decode_loss.item()

            avg_acc = tot_acc / (bid + 1)
            avg_real_acc = tot_real_acc / (bid + 1)
            avg_loss = tot_loss / (bid + 1)
            avg_warp_loss = tot_warp_loss / (bid + 1)
            avg_decode_loss = tot_decode_loss / (bid + 1)

            progress.set_description(
                f'| epoch {eid:03d} | acc {avg_acc:.4f} '
                f'| r_acc {avg_real_acc:.4f} | loss {avg_loss:.4f} '
                f'| warp {avg_warp_loss:.4f} | decode {avg_decode_loss:.4f} |')

        # for wmid, vs in var_selection.items():
        #     self.logger.info(f'wmid: {wmid} | vs: {Counter(vs)}')

        res_dict = {
            'epoch': eid,
            'acc': avg_acc,
            'real_acc': avg_real_acc,
            'loss': avg_loss,
            'warp_loss': avg_warp_loss,
            'decode_loss': avg_decode_loss,
        }

        return res_dict

    def _test_epoch(self, eid: int, dataloader: DataLoader):
        self.code_encoder.eval()
        self.selector.eval()
        self.approximator.eval()
        self.wm_encoder.eval()
        self.wm_decoder.eval()
        if self.extract_encoder is not None:
            self.extract_encoder.eval()

        tot_loss = 0.0
        tot_acc = 0.0
        tot_oracle_acc = 0.0
        tot_warp_loss = 0.0
        tot_msg_acc = 0.0
        n_samples = 0

        for bid, batch in enumerate(dataloader):
            (x, length, padding_mask, instance_ids, wms, wmids) = batch
            B = x.shape[0]
            wm_len = wms.shape[1]

            n_samples += B

            x = x.to(self.device)
            wms = wms.float().to(self.device)
            padding_mask = padding_mask.to(self.device)
            wmids = wmids.to(self.device)

            # get style masks
            s_feasible = self.transform_manager.get_feasible_transform_ids(instance_ids)
            N_TRANS = self.transform_manager.get_transform_capacity()

            s_feasible_01 = []

            for s_f in s_feasible:
                val = torch.ones(N_TRANS, device=self.device).bool()
                val[s_f] = False
                s_feasible_01.append(val)
            s_masks = torch.stack(s_feasible_01, dim=0)

            # simulated encoding process
            code_feature = self.code_encoder(x, length, padding_mask)
            wm_feature = self.wm_encoder(wms)

            vs_output = self.selector.var_selector_forward(
                code_feature, wm_feature, random_mask=self.var_random_mask)
            vs_logits = torch.log_softmax(vs_output, dim=1)
            vs_onehots = F.gumbel_softmax(vs_logits, tau=0.5, hard=True)
            vs_ids = torch.argmax(vs_onehots, dim=1)

            ss_output = self.selector.transform_selector_forward(code_feature,
                                                                 wm_feature,
                                                                 transform_mask=s_masks)
            ss_logits = torch.log_softmax(ss_output, dim=1)
            ss_onehots = F.gumbel_softmax(ss_logits, tau=0.5, hard=True)
            ss_ids = torch.argmax(ss_onehots, dim=1)

            vs_embds = torch.matmul(vs_onehots, self.code_encoder.embedding.weight)
            ss_embds = self.approximator.get_transform_embedding(ss_onehots)
            feat_warped = self.approximator(code_feature, vs_embds, ss_embds,
                                            padding_mask)

            instances = self.transform_manager.get_transformed_codes_by_pred(
                instance_ids, ss_ids.tolist())
            instances, _ = self.transform_manager.varname_transform_on_instances(
                instances, vs_ids.tolist())

            # simulated decoding process
            xx, ll, mm = self.transform_manager.load_to_tensor(instances)
            xx = xx.to(self.device)
            mm = mm.to(self.device)

            if self.extract_encoder is not None:
                t_features = self.extract_encoder(xx, ll, mm)
            else:
                t_features = self.code_encoder(xx, ll, mm)
            outputs = self.wm_decoder(t_features)
            probs = torch.sigmoid(outputs)

            loss = self.loss_fn(probs, wms)
            preds = torch.where(probs > 0.5, 1, 0)
            tot_msg_acc += compute_msg_acc(preds, wms, wm_len)

            # oracle decoding process
            oracle_outputs = self.wm_decoder(feat_warped)
            oracle_probs = torch.sigmoid(oracle_outputs)
            oracle_preds = torch.where(oracle_probs > 0.5, 1, 0)

            # warp loss
            warp_loss = 0.1 * self.warper_loss_fn(t_features, feat_warped)

            tot_loss += loss.item() * B
            tot_warp_loss += warp_loss.item() * B
            tot_acc += ((preds == wms).float().sum().item()) / wm_len
            tot_oracle_acc += ((oracle_preds == wms).float().sum().item()) / wm_len

        avg_loss = tot_loss / n_samples
        avg_warp_loss = tot_warp_loss / n_samples
        avg_acc = tot_acc / n_samples
        avg_oracle_acc = tot_oracle_acc / n_samples
        avg_msg_acc = tot_msg_acc / n_samples

        return {
            'epoch': eid,
            'oracle_acc': avg_oracle_acc,
            'actual_acc': avg_acc,
            'loss': avg_loss,
            'warp_loss': avg_warp_loss,
            'msg_acc': avg_msg_acc,
        }

    def _save_models(self, save_fname: str):
        torch.save(
            {
                'model':
                self.code_encoder.state_dict(),
                'wm_encoder':
                self.wm_encoder.state_dict(),
                'wm_decoder':
                self.wm_decoder.state_dict(),
                'selector':
                self.selector.state_dict(),
                'approximator':
                self.approximator.state_dict(),
                'extract_encoder': (self.extract_encoder.state_dict()
                                    if self.extract_encoder is not None else None),
                'vocab':
                self.transform_manager.vocab,
            }, save_fname)

    def _pprint_res_dict(self, res: Dict, prefix: str = None) -> str:
        res_str = '|'
        for k, v in res.items():
            if isinstance(v, float):
                res_str += f' {k}: {v:.4f} |'
            elif isinstance(v, int):
                res_str += f' {k}: {v:3d} |'
            else:
                res_str += f' {k}: {v} |'

        if prefix is not None:
            assert isinstance(prefix, str), 'prefix must be a string'
            res_str = prefix + res_str

        return res_str

    def _new_best_metric(self, eval_res: Dict):
        metric = eval_res['actual_acc']
        if metric > self.best_metric:
            self.best_metric = metric
            return True
        return False

    def _post_eval_actions(self, eid: int, eval_res: Dict):
        self.logger.info(self._pprint_res_dict(eval_res, prefix='| valid '))
        if self._new_best_metric(eval_res):
            self._save_models(f'{self.save_dir}/models_best.pt')
            self.logger.info(f'| best model saved at {eid} epoch |')

    def _post_train_actions(self, eid: int, train_res: Dict):
        self.logger.info(self._pprint_res_dict(train_res, prefix='| train '))
        if self.scheduler is not None:
            self.scheduler.step()

        if eid == 24 or eid == 49:
            # save at 25th and 50th epoch
            self._save_models(f'{self.save_dir}/models_{eid}.pt')

    def do_train(self, num_epochs: int):
        try:
            for eid in range(num_epochs):
                train_res = self._train_epoch(eid, self.train_loader)
                self._post_train_actions(eid, train_res)

                with torch.no_grad():
                    valid_res = self._test_epoch(eid, self.valid_loader)
                    self._post_eval_actions(eid, valid_res)

                    test_res = self._test_epoch(eid, self.test_loader)
                    self.logger.info(self._pprint_res_dict(test_res, prefix='| test  '))
        except KeyboardInterrupt:
            self.logger.warn('interrupted')
            self._save_models(f'{self.save_dir}/models_backup.pt')
            return
        except Exception as e:
            self.logger.error(e)
            raise e
