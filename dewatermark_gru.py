import os
import json
import torch
import random
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR

from logger_setup import setup_logger_for_dewatermarking
from models import Seq2SeqAttentionGRU
from data_processing import CodeVocab
from data_processing import JsonlDewatermarkingDatasetProcessor, DewatermarkingCollator
from models.dewatermarker import create_mask, generate_square_subsequent_mask

from typing import Dict


def parse_args_for_adaptive_removal():
    parser = ArgumentParser()
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--dataset',
                        choices=[
                            'github_c_funcs', 'github_java_funcs', 'csn_java', 'csn_js',
                            'mbjp', 'mbjsp'
                        ],
                        default='github_java_funcs')
    parser.add_argument('--lang', choices=['cpp', 'java', 'javascript'], default='java')
    parser.add_argument('--dataset_dir',
                        type=str,
                        default='./datasets/detection/github_java_funcs')

    parser.add_argument('--log_prefix', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--model_arch', choices=['gru', 'transformer'], default='gru')
    parser.add_argument('--scheduler', action='store_true')

    parser.add_argument('--do_attack', action='store_true')
    parser.add_argument('--attack_checkpoint',
                        type=str,
                        default='./ckpts/dewatermark_gru')
    parser.add_argument('--attack_dataset_dir',
                        type=str,
                        default='./datasets/detection/github_java_funcs')

    return parser.parse_args()


def main():
    args = parse_args_for_adaptive_removal()
    N_EPOCHS = args.epochs
    LANG = args.lang
    DATASET_DIR = args.dataset_dir
    ATTACK_DATASET_DIR = args.attack_dataset_dir
    SEED = args.seed
    BATCH_SIZE = args.batch_size
    ATTACK_CHECKPOINT = args.attack_checkpoint
    DATASET = args.dataset

    logger = setup_logger_for_dewatermarking(args)
    logger.info(args)
    device = torch.device('cuda')

    # seed everything
    torch.manual_seed(SEED)
    random.seed(SEED)

    dataset_processor = JsonlDewatermarkingDatasetProcessor(lang=LANG)
    save_dir_name = f'./ckpts/dewatermark_gru_{args.log_prefix}'
    attack_output_filename = f'{save_dir_name}/{os.path.basename(ATTACK_DATASET_DIR)}_{DATASET}.json'
    if not os.path.exists(save_dir_name):
        os.makedirs(save_dir_name)

    if args.do_train:
        # load datasets
        logger.info('Processing original dataset')
        instance_dict = dataset_processor.load_jsonls(DATASET_DIR)
        train_instances = instance_dict['train']
        valid_instances = instance_dict['valid']
        test_instances = instance_dict['test']

        vocab = dataset_processor.build_vocab(train_instances)

        train_dataset = dataset_processor.build_dataset(train_instances, vocab)
        valid_dataset = dataset_processor.build_dataset(valid_instances, vocab)
        test_dataset = dataset_processor.build_dataset(test_instances, vocab)
        logger.info(f'Vocab size: {len(vocab)}')
        logger.info(f'Train size: {len(train_dataset)}')
        logger.info(f'Test size: {len(test_dataset)}')

        FEATURE_DIM = 512
        model = Seq2SeqAttentionGRU(vocab_size=len(vocab),
                                    hidden_size=FEATURE_DIM,
                                    bos_idx=vocab.bos_idx())
        initial_lr = 1e-3
        model.to(device)
        optimizer = optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=1e-2)
        # scheduler = ExponentialLR(optimizer, gamma=0.85)
        scheduler = None

        loss_fn = nn.NLLLoss(ignore_index=vocab.pad_idx())

        logger.info('Starting training loop')
        logger.info('Constructing dataloaders and models')
        train_loader = DataLoader(train_dataset,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True,
                                  collate_fn=DewatermarkingCollator(batch_first=True))
        valid_loader = DataLoader(valid_dataset,
                                  batch_size=BATCH_SIZE,
                                  shuffle=False,
                                  collate_fn=DewatermarkingCollator(batch_first=True))
        test_loader = DataLoader(test_dataset,
                                 batch_size=BATCH_SIZE,
                                 shuffle=False,
                                 collate_fn=DewatermarkingCollator(batch_first=True))
        # training loop
        try:
            best_loss = float('inf')
            for eid in range(N_EPOCHS):
                train_res = train(eid, train_loader, model, loss_fn, optimizer, vocab,
                                  device)
                logger.info(pprint_res_dict(train_res, prefix='| train '))

                if scheduler is not None:
                    scheduler.step()

                valid_res, decoded_text = evaluate(eid,
                                                   valid_loader,
                                                   model,
                                                   loss_fn,
                                                   vocab,
                                                   device,
                                                   sample=True)
                logger.info(pprint_res_dict(valid_res, prefix='| valid '))

                for i, (src, tgt, pred) in enumerate(decoded_text):
                    logger.info(f'Sample#{i}')
                    logger.info(f'    Source   : {src}')
                    logger.info(f'    Target   : {tgt}')
                    logger.info(f'    Generated: {pred}')

                # check if should save
                if valid_res['loss'] < best_loss:
                    with open(f'{save_dir_name}/best_model.pt', 'wb') as fo:
                        torch.save({'model': model.state_dict(), 'vocab': vocab}, fo)
                    logger.info(f'| best model saved at {eid} epoch |')
                    best_loss = valid_res['loss']

                test_res = evaluate(eid, test_loader, model, loss_fn, vocab, device)
                logger.info(pprint_res_dict(test_res, prefix='| test  '))
        except KeyboardInterrupt:
            logger.info('KeyboardInterrupt.')

    # load best model
    if args.do_attack:
        print('processing attack dataset')
        attack_instances = dataset_processor.load_jsonl(ATTACK_DATASET_DIR, split='test')

        print('loading best model')
        with open(ATTACK_CHECKPOINT, 'rb') as fo:
            save_dict = torch.load(fo)
        vocab: CodeVocab = save_dict['vocab']

        FEATURE_DIM = 512
        model = Seq2SeqAttentionGRU(vocab_size=len(vocab),
                                    hidden_size=FEATURE_DIM,
                                    bos_idx=vocab.bos_idx())

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'Model parameters: {n_params:,}')

        initial_lr = 1e-3
        model.to(device)
        model.load_state_dict(save_dict['model'])
        attack_dataset = dataset_processor.build_dataset(attack_instances, vocab)
        attack_loader = DataLoader(attack_dataset,
                                   batch_size=1,
                                   shuffle=False,
                                   collate_fn=DewatermarkingCollator(batch_first=True))

        dewatermark_output = dewatermark_procedure(attack_loader, model, vocab, device)
        with open(attack_output_filename, 'w') as fo:
            json.dump(dewatermark_output, fo, indent=4, ensure_ascii=False)
        print(f'attack output saved to {attack_output_filename}')


def pprint_res_dict(res: Dict, prefix: str = None) -> str:
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


def train(eid: int, dataloader: DataLoader, model: Seq2SeqAttentionGRU,
          loss_fn: nn.Module, optimizer: optim.Optimizer, vocab: CodeVocab,
          device: torch.device):
    model.train()

    progress = tqdm(dataloader)
    tot_loss = 0.0
    for bid, batch in enumerate(progress):
        optimizer.zero_grad()
        src_ids, tgt_ids = batch
        src_ids = src_ids.to(device)
        tgt_ids = tgt_ids.to(device)

        decoder_outputs = model(src_ids, tgt_ids)
        loss = loss_fn(decoder_outputs.view(-1, decoder_outputs.shape[-1]),
                       tgt_ids[:, 1:].contiguous().view(-1))

        loss.backward()
        optimizer.step()

        tot_loss += loss.item()
        avg_loss = tot_loss / (bid + 1)
        progress.set_description(f'| epoch {eid:03d} | loss {avg_loss:.4f} |')

    return {'epoch': eid, 'loss': avg_loss}


def evaluate(eid: int,
             dataloader: DataLoader,
             model: nn.Module,
             loss_fn: nn.Module,
             vocab: CodeVocab,
             device: torch.device,
             sample: bool = False):
    model.eval()

    progress = tqdm(dataloader)
    tot_loss = 0.0
    with torch.no_grad():
        for bid, batch in enumerate(progress):
            src_ids, tgt_ids = batch
            src_ids = src_ids.to(device)
            tgt_ids = tgt_ids.to(device)

            decoder_outputs = model(src_ids, tgt_ids)
            loss = loss_fn(decoder_outputs.view(-1, decoder_outputs.shape[-1]),
                           tgt_ids[:, 1:].contiguous().view(-1))

            tot_loss += loss.item()
            avg_loss = tot_loss / (bid + 1)
            progress.set_description(f'| epoch {eid:03d} | loss {avg_loss:.4f} |')

            if sample and bid == 0:
                samples = []
                for i in range(5):
                    _, topi = decoder_outputs[i].topk(1)
                    decoded_ids = topi.squeeze()  # L

                    decoded_words = []
                    for idx in decoded_ids:
                        decoded_words.append(vocab.get_token_by_id(idx.item()))
                        if idx == vocab.eos_idx():
                            break
                    src_words = []
                    for idx in src_ids[i]:
                        src_words.append(vocab.get_token_by_id(idx.item()))
                        if idx == vocab.eos_idx():
                            break
                    tgt_words = []
                    for idx in tgt_ids[i]:
                        tgt_words.append(vocab.get_token_by_id(idx.item()))
                        if idx == vocab.eos_idx():
                            break

                    src_sent = ' '.join(src_words)
                    tgt_sent = ' '.join(tgt_words)
                    decoded = ' '.join(decoded_words)
                    samples.append((src_sent, tgt_sent, decoded))

    if sample:
        return {'epoch': eid, 'loss': avg_loss}, samples
    else:
        return {'epoch': eid, 'loss': avg_loss}


def greedy_decode(model: Seq2SeqAttentionGRU, src: torch.Tensor, tgt: torch.Tensor,
                  vocab: CodeVocab, device: torch.device):
    model.eval()
    src = src.to(device)
    tgt = tgt.to(device)

    assert src.shape[0] == 1

    decoder_outputs = model(src, None)

    _, topi = decoder_outputs.topk(1)
    decoded_ids = topi.squeeze()  # L

    decoded_words = []
    for idx in decoded_ids.tolist():
        decoded_words.append(vocab.get_token_by_id(idx))
        if idx == vocab.eos_idx():
            break
    decoded_words = decoded_words[:-1]  # remove eos

    return ' '.join(decoded_words), decoded_words


def dewatermark_procedure(dataloader: DataLoader, model: Seq2SeqAttentionGRU,
                          vocab: CodeVocab, device: torch.device):
    model.eval()

    rewatermarked_samples = []
    progress = tqdm(dataloader)
    progress.set_description('| dewatermarking |')
    with torch.no_grad():
        for _, batch in enumerate(progress):
            (src_ids, tgt_ids) = batch
            sentence, words = greedy_decode(model, src_ids, tgt_ids, vocab, device)
            rewatermarked_samples.append({'sentence': sentence, 'tokens': words})

    return rewatermarked_samples


if __name__ == '__main__':
    main()
