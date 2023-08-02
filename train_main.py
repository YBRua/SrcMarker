import torch
import random
import tree_sitter
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from argparse import ArgumentParser
from torch.optim.lr_scheduler import ExponentialLR

from logger_setup import setup_logger
from models import (
    ConcatApproximator,
    TransformationApproximator,
    VarApproximator,
    TransformSelector,
    TransformerEncoderExtractor,
    GRUEncoder,
    ExtractGRUEncoder,
    WMLinearEncoder,
    MLP2,
)

from code_transform_provider import CodeTransformProvider
from runtime_data_manager import InMemoryJitRuntimeDataManager
from trainers import UltimateWMTrainer, UltimateVarWMTrainer, UltimateTransformTrainer
from data_processing import JsonlWMDatasetProcessor, DynamicWMCollator
import mutable_tree.transformers as ast_transformers


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument(
        '--dataset',
        choices=['github_c_funcs', 'github_java_funcs', 'csn_java', 'csn_js'],
        default='github_c_funcs')
    parser.add_argument('--lang', choices=['cpp', 'java', 'javascript'], default='c')
    parser.add_argument('--dataset_dir', type=str, default='./datasets/github_c_funcs')

    parser.add_argument('--log_prefix', type=str, default='')
    parser.add_argument('--n_bits', type=int, default=4)
    parser.add_argument('--style_only', action='store_true')
    parser.add_argument('--var_only', action='store_true')
    parser.add_argument('--var_nomask', action='store_true')
    parser.add_argument('--varmask_prob', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--wv', type=float, default=0.5)
    parser.add_argument('--wt', type=float, default=0.5)
    parser.add_argument('--model_arch', choices=['gru', 'transformer'], default='gru')
    parser.add_argument('--scheduler', action='store_true')
    parser.add_argument('--shared_encoder', action='store_true')
    parser.add_argument('--var_transform_mode',
                        choices=['replace', 'append'],
                        default='replace')

    return parser.parse_args()


def main():
    args = parse_args()
    N_EPOCHS = args.epochs
    LANG = args.lang
    DATASET = args.dataset
    DATASET_DIR = args.dataset_dir
    N_BITS = args.n_bits
    SEED = args.seed
    BATCH_SIZE = args.batch_size
    MODEL_ARCH = args.model_arch
    SHARED_ENCODER = args.shared_encoder
    VARMASK_PROB = args.varmask_prob
    VAR_TRANSFORM_MODE = args.var_transform_mode

    logger = setup_logger(args)
    logger.info(args)
    device = torch.device('cuda')

    # seed everything
    torch.manual_seed(SEED)
    random.seed(SEED)

    # load datasets
    dataset_processor = JsonlWMDatasetProcessor(lang=LANG)
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
    all_instances = train_instances + valid_instances + test_instances

    # initialize transform computers
    parser = tree_sitter.Parser()
    parser_lang = tree_sitter.Language('./parser/languages.so', LANG)
    parser.set_language(parser_lang)
    code_transformers = [
        ast_transformers.IfBlockSwapTransformer(),
        ast_transformers.CompoundIfTransformer(),
        ast_transformers.ConditionTransformer(),
        ast_transformers.LoopTransformer(),
        ast_transformers.InfiniteLoopTransformer(),
        ast_transformers.UpdateTransformer(),
        ast_transformers.SameTypeDeclarationTransformer(),
        ast_transformers.VarDeclLocationTransformer(),
        ast_transformers.VarInitTransformer(),
        ast_transformers.VarNameStyleTransformer()
    ]
    transform_computer = CodeTransformProvider(LANG, parser, code_transformers)
    transform_manager = InMemoryJitRuntimeDataManager(transform_computer,
                                                      all_instances,
                                                      lang=LANG)
    transform_manager.register_vocab(vocab)
    transform_manager.load_transform_mask(f'datasets/feasible_transform_{DATASET}.json')
    transform_manager.load_varname_dict(f'datasets/variable_names_{DATASET}.json')
    transform_capacity = transform_manager.get_transform_capacity()

    logger.info('Constructing dataloaders and models')
    train_loader = DataLoader(train_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              collate_fn=DynamicWMCollator(N_BITS))
    valid_loader = DataLoader(valid_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=False,
                              collate_fn=DynamicWMCollator(N_BITS))
    test_loader = DataLoader(test_dataset,
                             batch_size=BATCH_SIZE,
                             shuffle=False,
                             collate_fn=DynamicWMCollator(N_BITS))

    logger.info(f'Using {MODEL_ARCH}')
    EMBEDDING_DIM = 768
    FEATURE_DIM = 768
    if MODEL_ARCH == 'gru':
        encoder = GRUEncoder(vocab_size=len(vocab),
                             hidden_size=FEATURE_DIM,
                             embedding_size=EMBEDDING_DIM)
        if SHARED_ENCODER:
            extract_encoder = None
        else:
            extract_encoder = ExtractGRUEncoder(vocab_size=len(vocab),
                                                hidden_size=FEATURE_DIM,
                                                embedding_size=EMBEDDING_DIM)
        encoder_lr = 1e-3
    elif MODEL_ARCH == 'transformer':
        encoder = TransformerEncoderExtractor(vocab_size=len(vocab),
                                              embedding_size=EMBEDDING_DIM,
                                              hidden_size=FEATURE_DIM)
        if SHARED_ENCODER:
            extract_encoder = None
        else:
            extract_encoder = TransformerEncoderExtractor(vocab_size=len(vocab),
                                                          embedding_size=EMBEDDING_DIM,
                                                          hidden_size=FEATURE_DIM)
        encoder_lr = 3e-4
        if not args.scheduler:
            encoder_lr = 5e-5
    logger.info(f'learning rate: {encoder_lr}')

    if VAR_TRANSFORM_MODE == 'replace':
        vmask = vocab.get_valid_identifier_mask()
    else:
        vmask = vocab.get_valid_highfreq_mask(32 * 2**N_BITS)

    selector = TransformSelector(vocab_size=len(vocab),
                                 transform_capacity=transform_capacity,
                                 input_dim=FEATURE_DIM,
                                 vocab_mask=vmask,
                                 random_mask_prob=VARMASK_PROB)

    if args.var_only:
        approximator = VarApproximator(vocab_size=len(vocab),
                                       input_dim=FEATURE_DIM,
                                       output_dim=FEATURE_DIM)
    elif args.style_only:
        approximator = TransformationApproximator(transform_capacity=transform_capacity,
                                                  input_dim=FEATURE_DIM,
                                                  output_dim=FEATURE_DIM)
    else:
        approximator = ConcatApproximator(vocab_size=len(vocab),
                                          transform_capacity=transform_capacity,
                                          input_dim=FEATURE_DIM,
                                          output_dim=FEATURE_DIM)
    logger.info(f'approximator arch: {approximator.__class__.__name__}')

    wm_encoder = WMLinearEncoder(N_BITS, embedding_dim=FEATURE_DIM)
    wm_decoder = MLP2(output_dim=N_BITS, bn=False, input_dim=FEATURE_DIM)

    encoder.to(device)
    if extract_encoder is not None:
        extract_encoder.to(device)
    selector.to(device)
    approximator.to(device)
    wm_encoder.to(device)
    wm_decoder.to(device)

    if SHARED_ENCODER:
        scheduled_optim = optim.AdamW([
            {
                'params': encoder.parameters(),
                'lr': encoder_lr,
                'weight_decay': 0.01,
            },
        ])
    else:
        scheduled_optim = optim.AdamW([
            {
                'params': encoder.parameters(),
                'lr': encoder_lr,
                'weight_decay': 0.01,
            },
            {
                'params': extract_encoder.parameters(),
                'lr': encoder_lr,
                'weight_decay': 0.01,
            },
        ])
    other_optim = optim.Adam([
        {
            'params': selector.parameters()
        },
        {
            'params': approximator.parameters()
        },
        {
            'params': wm_encoder.parameters()
        },
        {
            'params': wm_decoder.parameters()
        },
    ])

    if args.scheduler:
        logger.info('Using exponential scheduler')
        scheduler = ExponentialLR(scheduled_optim, gamma=0.85)
    else:
        scheduler = None

    loss_fn = nn.BCELoss()

    logger.info('Starting training loop')
    save_dir_name = f'{args.log_prefix}_{args.seed}_{args.dataset}'

    if args.style_only:
        logger.info('[ABLATION] Style only')
        trainer = UltimateTransformTrainer(code_encoder=encoder,
                                           extract_encoder=extract_encoder,
                                           wm_encoder=wm_encoder,
                                           selector=selector,
                                           approximator=approximator,
                                           wm_decoder=wm_decoder,
                                           scheduled_optimizer=scheduled_optim,
                                           other_optimizer=other_optim,
                                           device=device,
                                           train_loader=train_loader,
                                           valid_loader=valid_loader,
                                           test_loader=test_loader,
                                           loss_fn=loss_fn,
                                           transform_manager=transform_manager,
                                           scheduler=scheduler,
                                           logger=logger,
                                           ckpt_dir=save_dir_name)
    elif args.var_only:
        logger.info('[ABLATION] Var only')
        trainer = UltimateVarWMTrainer(code_encoder=encoder,
                                       extract_encoder=extract_encoder,
                                       wm_encoder=wm_encoder,
                                       selector=selector,
                                       approximator=approximator,
                                       wm_decoder=wm_decoder,
                                       scheduled_optimizer=scheduled_optim,
                                       other_optimizer=other_optim,
                                       device=device,
                                       train_loader=train_loader,
                                       valid_loader=valid_loader,
                                       test_loader=test_loader,
                                       loss_fn=loss_fn,
                                       transform_manager=transform_manager,
                                       scheduler=scheduler,
                                       logger=logger,
                                       ckpt_dir=save_dir_name)
    else:
        trainer = UltimateWMTrainer(encoder,
                                    extract_encoder,
                                    wm_encoder,
                                    selector,
                                    approximator,
                                    wm_decoder,
                                    scheduled_optim,
                                    other_optim,
                                    device,
                                    train_loader,
                                    valid_loader,
                                    test_loader,
                                    loss_fn,
                                    transform_manager,
                                    w_var=args.wv,
                                    w_style=args.wt,
                                    scheduler=scheduler,
                                    logger=logger,
                                    ckpt_dir=save_dir_name,
                                    var_transform_mode=VAR_TRANSFORM_MODE)
        trainer.set_var_random_mask_enabled(not args.var_nomask)
        logger.info(f'w_var: {args.wv}, w_style: {args.wt}')
        logger.info(f'var mask enabled: {not args.var_nomask}')

    trainer.do_train(N_EPOCHS)


if __name__ == '__main__':
    main()
