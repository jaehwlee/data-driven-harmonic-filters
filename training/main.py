import os
import argparse
from solver import Solver

def main(config):
    assert config.dataset in {'mtat', 'dcase', 'keyword', 'pandora'},\
        'invalid mode: "{}" not in ["mtat", "dcase", "keyword", "pandora"]'.format(config.task)

    # path for models
    if not os.path.exists(config.model_save_path):
        os.makedirs(config.model_save_path)

    # import data loader
    if config.dataset == 'mtat':
        from data_loader.mtat_loader import get_audio_loader
    elif config.dataset == 'dcase':
        from data_loader.dcase_loader import get_audio_loader
    elif config.dataset == 'keyword':
        from data_loader.keyword_loader import get_audio_loader

    # get data loder
    data_loader = get_audio_loader(config.data_path,
                                       config.batch_size,
                                       input_length=config.input_length,
                                       num_workers=config.num_workers)

    solver = Solver(data_loader, config)

    print('stage1_train')
    solver.stage1_train()
    print('stage2_train')
    solver.stage2_train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # model hyper-parameters
    parser.add_argument('--conv_channels', type=int, default=128)
    parser.add_argument('--sample_rate', type=int, default=16000)
    parser.add_argument('--n_fft', type=int, default=512)
    parser.add_argument('--n_harmonic', type=int, default=6)
    parser.add_argument('--semitone_scale', type=int, default=2)
    parser.add_argument('--learn_bw', type=str, default='only_Q', choices=['only_Q', 'fix'])

    # dataset
    parser.add_argument('--input_length', type=int, default=80000)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--dataset', type=str, default='mtat', choices=['mtat', 'dcase', 'keyword'])

    # training settings
    parser.add_argument('--n_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--use_tensorboard', type=int, default=1)
    parser.add_argument('--model_save_path', type=str, default='./../models')
    parser.add_argument('--model_load_path', type=str, default='.')
    parser.add_argument('--data_path', type=str, default='../../tf2-music-tagging-models/dataset')
    parser.add_argument('--log_step', type=int, default=20)

    config = parser.parse_args()

    print(config)
    main(config)
