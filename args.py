import argparse

def get_train_args():
    parser = argparse.ArgumentParser('Train a model on SQuAD')

    add_common_args(parser)
    add_train_test_args(parser)

    parser.add_argument('--eval_steps',
                        type=int,
                        default=50000,
                        help='Number of steps between successive evaluations')
    parser.add_argument('--lr',
                        type=float,
                        default=0.5,
                        help='Learning rate')
    parser.add_argument('--l2_wd',
                        type=float,
                        default=0.0,
                        help='L2 weight decay')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=30,
                        help='Number of epochs for which to train. Negative means forever')
    parser.add_argument('--drop_prob',
                        type=float,
                        default=0.2,
                        help='Probability of zeroing an activation in dripout layers')
    parser.add_argument('--metric_name',
                        type=str,
                        default='F1',
                        choices=('NLL', 'EM', 'F1'),
                        help='Name of dev metric to determine best checkpoint')
    parser.add_argument('--max_checkpoints',
                        type=int,
                        default=5,
                        help='Maximum number of checkpoints to keep on disk')
    parser.add_argument('--max_grad_norm',
                        type=float,
                        default=5.0,
                        help='Maximum gradient norm for gradient clipping')
    parser.add_argument('--seed',
                        type=int,
                        default=224,
                        help='Random seed for reproducibility')
    parser.add_argument('--ema_decay',
                        type=float,
                        default=0.999,
                        help='Decay rate for exponential moving average of parameters')
    
    args = parser.parse_args()

    if args.metric_name == 'NLL':
        args.maximize_metric = False
    elif args.metric_name in ('EM', 'F1'):
        args.maximize_metric = True
    else:
        raise ValueError(f'Unrecognized metric name: "{args.metric_name}"')
    
    return args


def get_test_args():
    parser = argparse.ArgumentParser('Test a trained model on SQuAD')

    add_common_args(parser)
    add_train_test_args(parser)

    parser.add_argument('--split',
                        type=str,
                        default='dev',
                        choices=('train', 'dev', 'test'),
                        help='Split to use for testing.')
    parser.add_argument('--sub_file',
                        type=str,
                        default='submission.csv',
                        help='Name for submission file.')

    # Require load_path for test.py
    args = parser.parse_args()
    if not args.load_path:
        raise argparse.ArgumentError('Missing required argument --load_path')

    return args


def add_common_args(parser):
    parser.add_argument('--train_record_file',
                        type=str,
                        default='./data/train.npz')
    parser.add_argument('--dev_record_file',
                        type=str,
                        default='./data/dev.npz')
    parser.add_argument('--test_record_file',
                        type=str,
                        default='./data/test.npz')
    parser.add_argument('--word_emb_file',
                        type=str,
                        default='./data/word_emb.json')
    parser.add_argument('--char_emb_file',
                        type=str,
                        default='./data/char_emb.json')
    parser.add_argument('--train_eval_file',
                        type=str,
                        default='./data/train_eval.json')
    parser.add_argument('--dev_eval_file',
                        type=str,
                        default='./data/dev_eval.json')
    parser.add_argument('--test_eval_file',
                        type=str,
                        default='./data/test_eval.json')


def add_train_test_args(parser):
    parser.add_argument('--name',
                        '-n',
                        type=str,
                        required=True,
                        help='Name to identify training or test run')
    parser.add_argument('--max_ans_len',
                        type=int,
                        default=15,
                        help='Maximum length of a predicted answer')
    parser.add_argument('--num_workers',
                        type=int,
                        default=4,
                        help='Number of sub-processes to use per data loader')
    parser.add_argument('--base_dir',
                        type=str,
                        default='./output/',
                        help='Base directory for saving information')
    parser.add_argument('--batch_size',
                        type=int,
                        default=64,
                        help='Batch size')
    parser.add_argument('--use_squad_v2',
                        type=lambda s: s.lower().startswith('t'),
                        default=True,
                        help='Whether to use SQuAD 2.0 (unanswerable) questions')
    parser.add_argument('--hidden_size',
                        type=int,
                        default=100,
                        help='Number of features in encoder hidden layers')
    parser.add_argument('--num_visuals',
                        type=int,
                        default=10,
                        help='Number of examples to visualize in Tensorboard')
    parser.add_argument('--load_path',
                        type=str,
                        default=None,
                        help='Path to load as a model checkpoint')
    parser.add_argument('--n_highway_layers',
                        type=int,
                        default=2,
                        help='Number of layers in highway network')