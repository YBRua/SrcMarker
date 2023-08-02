import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import logging

from models import (TransformSelector, DecodeLossApproximator)

from tqdm import tqdm
from torch.utils.data import DataLoader
from eval_utils import compute_msg_acc

from logger_setup import DefaultLogger
from runtime_data_manager import InMemoryJitRuntimeDataManager
from typing import List, Dict, Optional

from collections import defaultdict, Counter


class ActorCriticWMTrainer:
    def __init__(self,
                 code_encoder: nn.Module,
                 wm_encoder: nn.Module,
                 selector: TransformSelector,
                 approximator: DecodeLossApproximator,
                 wm_decoder: nn.Module,
                 actor_optim: optim.Optimizer,
                 critic_optim: optim.Optimizer,
                 decoder_optim: optim.Optimizer,
                 device: torch.device,
                 train_loader: DataLoader,
                 valid_loader: DataLoader,
                 test_loader: DataLoader,
                 transform_manager: InMemoryJitRuntimeDataManager,
                 scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
                 logger: Optional[logging.Logger] = None,
                 ckpt_dir: str = 'my_model'):

        self.code_encoder = code_encoder
        self.selector = selector
        self.critic = approximator
        self.wm_encoder = wm_encoder
        self.wm_decoder = wm_decoder

        self.actor_optim = actor_optim
        self.critic_optim = critic_optim
        self.decoder_optim = decoder_optim

        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

        self.transform_manager = transform_manager

        self.critic_loss_fn = nn.MSELoss()
        self.wm_loss_fn = nn.BCELoss(reduction='none')

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

    def _transform_sample(self, eid: int, actor_tprobs: torch.Tensor,
                          actor_vprobs: torch.Tensor, feasible_ts: List[List[int]],
                          feasible_v: List[int]):
        should_uniform = 1 / (eid + 1)
        rand = random.random()

        if rand < should_uniform:
            sampled_ts_ids = []
            sampled_vs_ids = []
            feasible_v_idx = feasible_v
            for feasible_t_idx in feasible_ts:
                sampled_ts_ids.append(random.choice(feasible_t_idx))
                sampled_vs_ids.append(random.choice(feasible_v_idx))
            return sampled_ts_ids, sampled_vs_ids
        else:
            sampled_ts_ids = torch.multinomial(actor_tprobs, 1).squeeze(1).tolist()
            sampled_vs_ids = torch.multinomial(actor_vprobs, 1).squeeze(1).tolist()
            return sampled_ts_ids, sampled_vs_ids

    def _train_epoch(self, eid: int, dataloader: DataLoader):
        # for debugging purposes
        var_selection = defaultdict(list)

        self.code_encoder.train()
        self.selector.train()
        self.critic.train()
        self.wm_encoder.train()
        self.wm_decoder.train()

        progress = tqdm(dataloader)
        tot_loss = 0.0
        tot_acc = 0.0
        tot_actor_loss = 0.0
        tot_critic_loss = 0.0
        tot_decoder_loss = 0.0

        for bid, batch in enumerate(progress):
            (x, length, padding_mask, instance_ids, wms, wmids) = batch
            B = x.size(0)

            x = x.to(self.device)
            wms = wms.float().to(self.device)
            padding_mask = padding_mask.to(self.device)
            wmids = wmids.to(self.device)

            # get style masks
            t_feasible = self.transform_manager.get_feasible_transform_ids(instance_ids)
            N_TRANS = self.transform_manager.get_transform_capacity()

            t_feasible_01 = []
            for t_f in t_feasible:
                val = torch.ones(N_TRANS, device=self.device).bool()
                val[t_f] = False
                t_feasible_01.append(val)
            t_masks = torch.stack(t_feasible_01, dim=0)

            self.actor_optim.zero_grad()
            self.critic_optim.zero_grad()
            self.decoder_optim.zero_grad()

            # original code feature
            code_feature = self.code_encoder(x, length, padding_mask)
            wm_feature = self.wm_encoder(wms)

            # select transforms
            # vs for variable selection, ts for transform selection
            vs_output = self.selector.var_selector_forward(code_feature,
                                                           wm_feature,
                                                           random_mask=False)
            ts_output = self.selector.transform_selector_forward(code_feature,
                                                                 wm_feature,
                                                                 t_masks,
                                                                 random_mask=False)
            vs_probs = torch.softmax(vs_output, dim=-1)
            ts_probs = torch.softmax(ts_output, dim=-1)

            # CRITIC UPDATE
            ts_probs_detach = ts_probs.detach().clone()
            vs_probs_detach = vs_probs.detach().clone()

            # sample transformations
            sampled_ts_ids, sampled_vs_ids = self._transform_sample(
                eid, ts_probs_detach, vs_probs_detach, t_feasible,
                self.transform_manager.vocab.get_valid_identifier_idx())
            sampled_instances = self.transform_manager.get_transformed_codes_by_pred(
                instance_ids, sampled_ts_ids)
            sampled_instances, _ = self.transform_manager.varname_transform_on_instances(
                sampled_instances, sampled_vs_ids)

            # transformed code feature
            sampled_x, sampled_l, sampled_mask = self.transform_manager.load_to_tensor(
                sampled_instances)
            sampled_x = sampled_x.to(self.device)
            sampled_mask = sampled_mask.to(self.device)
            sampled_features = self.code_encoder(sampled_x, sampled_l, sampled_mask)

            # compute decoding loss
            sampled_outputs = self.wm_decoder(sampled_features)
            sampled_probs = torch.sigmoid(sampled_outputs)
            sampled_decode_loss = self.wm_loss_fn(sampled_probs,
                                                  wms).detach().clone()  # B, n_bits

            # compute approximated decoding loss
            # B, 1, H. Additional 1 is to align with approximator forward() impl
            sampled_ts_ids = torch.tensor(sampled_ts_ids, device=self.device)
            sampled_vs_ids = torch.tensor(sampled_vs_ids, device=self.device)
            transform_embds_ = self.critic.t_embeddings(sampled_ts_ids.unsqueeze(1))
            variable_embds_ = self.code_encoder.embedding(sampled_vs_ids.unsqueeze(1))

            # B, 1, n_bits
            approx_loss_ = self.critic(code_feature, variable_embds_, transform_embds_)

            # if eid >= 2 and bid % 10 == 0:
            #     print('approx_loss', approx_loss_.squeeze(1))
            #     print('sampled_decode_loss', sampled_decode_loss)

            critic_loss = self.critic_loss_fn(approx_loss_.squeeze(1),
                                              sampled_decode_loss)

            # ACTOR UPDATE
            # sample from feasible transform and variable
            SAMPLE_SIZE = 30000
            transform_sample_idx = []
            variable_sample_idx = []
            valid_v_idx = self.transform_manager.vocab.get_valid_identifier_idx()
            valid_v_idx = torch.tensor(valid_v_idx, device=self.device)

            for valid_t_idx in t_feasible:
                valid_t_idx = torch.tensor(valid_t_idx, device=self.device)
                randidx = torch.randint(0,
                                        len(valid_t_idx), (SAMPLE_SIZE, ),
                                        device=self.device)
                transform_sample_idx.append(valid_t_idx[randidx])
                randidx = torch.randint(0,
                                        len(valid_v_idx), (SAMPLE_SIZE, ),
                                        device=self.device)
                variable_sample_idx.append(valid_v_idx[randidx])
            transform_sample_idx = torch.stack(transform_sample_idx, dim=0)
            variable_sample_idx = torch.stack(variable_sample_idx, dim=0)

            batch_idx = torch.arange(B, device=self.device)[..., None]
            vs_probs_sample = vs_probs[batch_idx, variable_sample_idx]
            ts_probs_sample = ts_probs[batch_idx, transform_sample_idx]

            # B, S
            joint_probs = vs_probs_sample * ts_probs_sample
            # B, S, H
            transform_embds = self.critic.t_embeddings(transform_sample_idx)
            variable_embds = self.code_encoder.embedding(variable_sample_idx)
            actor_loss = self.critic(code_feature, variable_embds,
                                     transform_embds).detach().clone()  # B, S, N
            actor_loss = torch.mean(actor_loss, dim=-1)  # B, S
            actor_loss = torch.sum(actor_loss * joint_probs, dim=-1)  # B
            actor_loss = torch.mean(actor_loss)  # reduction: mean

            # update actor critic
            loss = actor_loss + critic_loss
            loss.backward()
            self.actor_optim.step()
            self.critic_optim.step()

            # DECODER UPDATE
            self.actor_optim.zero_grad()
            self.critic_optim.zero_grad()
            self.decoder_optim.zero_grad()

            ts_ids = torch.argmax(ts_probs_detach, dim=-1)
            vs_ids = torch.argmax(vs_probs_detach, dim=-1)

            tinstances = self.transform_manager.get_transformed_codes_by_pred(
                instance_ids, ts_ids.tolist())
            tinstances, _ = self.transform_manager.varname_transform_on_instances(
                tinstances, vs_ids.tolist())
            tx, tl, tmask = self.transform_manager.load_to_tensor(tinstances)

            tx = tx.to(self.device)
            tmask = tmask.to(self.device)

            for wmid, vs in zip(wmids.tolist(), vs_ids.tolist()):
                word = self.transform_manager.vocab.get_token_by_id(vs)
                var_selection[wmid].append(word)

            tfeatures = self.code_encoder(tx, tl, tmask)
            tdecode = self.wm_decoder(tfeatures)

            tprobs = torch.sigmoid(tdecode)
            decode_loss = self.wm_loss_fn(tprobs, wms).reshape(-1).mean()
            decode_loss.backward()
            self.actor_optim.step()
            self.decoder_optim.step()

            # update metrics and avg losses
            preds = torch.where(tprobs > 0.5, 1, 0)
            tot_acc += (preds == wms).float().mean().item()
            tot_actor_loss += actor_loss.item()
            tot_critic_loss += critic_loss.item()
            tot_decoder_loss += decode_loss.reshape(-1).mean().item()
            tot_loss += (actor_loss.item() + critic_loss.item() +
                         decode_loss.reshape(-1).mean().item())

            avg_acc = tot_acc / (bid + 1)
            avg_actor_loss = tot_actor_loss / (bid + 1)
            avg_critic_loss = tot_critic_loss / (bid + 1)
            avg_decoder_loss = tot_decoder_loss / (bid + 1)
            avg_loss = tot_loss / (bid + 1)

            progress.set_description(
                f'| epoch {eid:03d} | acc {avg_acc:.4f} '
                f'| loss {avg_loss:.4f} | actor {avg_actor_loss:.4f} '
                f'| critic {avg_critic_loss:.4f} | decode {avg_decoder_loss:.4f} |')

        for wmid, vs in var_selection.items():
            self.logger.info(f'wmid: {wmid} | vs: {Counter(vs).most_common(10)}')

        res_dict = {
            'epoch': eid,
            'acc': avg_acc,
            'loss': avg_loss,
            'actor_loss': avg_actor_loss,
            'critic_loss': avg_critic_loss,
            'decode_loss': avg_decoder_loss,
        }

        return res_dict

    def _test_epoch(self, eid: int, dataloader: DataLoader):
        self.code_encoder.eval()
        self.selector.eval()
        self.critic.eval()
        self.wm_encoder.eval()
        self.wm_decoder.eval()

        tot_loss = 0.0
        tot_acc = 0.0
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

            instances = self.transform_manager.get_transformed_codes_by_pred(
                instance_ids, ss_ids.tolist())
            instances, _ = self.transform_manager.varname_transform_on_instances(
                instances, vs_ids.tolist())

            # simulated decoding process
            xx, ll, mm = self.transform_manager.load_to_tensor(instances)
            xx = xx.to(self.device)
            mm = mm.to(self.device)

            t_features = self.code_encoder(xx, ll, mm)
            outputs = self.wm_decoder(t_features)
            probs = torch.sigmoid(outputs)

            loss = self.wm_loss_fn(probs, wms)
            loss = loss.reshape(-1).mean()
            preds = torch.where(probs > 0.5, 1, 0)

            tot_loss += loss.item() * B
            tot_acc += ((preds == wms).float().sum().item()) / wm_len
            tot_msg_acc += compute_msg_acc(preds, wms, wm_len)

        avg_loss = tot_loss / n_samples
        avg_acc = tot_acc / n_samples
        avg_msg_acc = tot_msg_acc / n_samples

        return {
            'epoch': eid,
            'actual_acc': avg_acc,
            'loss': avg_loss,
            'msg_acc': avg_msg_acc,
        }

    def _save_models(self, save_fname: str):
        torch.save(
            {
                'model': self.code_encoder.state_dict(),
                'wm_encoder': self.wm_encoder.state_dict(),
                'wm_decoder': self.wm_decoder.state_dict(),
                'selector': self.selector.state_dict(),
                'approximator': self.critic.state_dict(),
                'extract_encoder': None,
                'vocab': self.transform_manager.vocab,
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
