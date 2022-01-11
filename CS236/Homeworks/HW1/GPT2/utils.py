import os
import argparse
import yaml
import matplotlib
if os.environ.get('DISPLAY', '') == '':
    matplotlib.use('Agg')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', action='store_true', help="Turn on GPU mode")

    args = parser.parse_args()
    return args


def dict2namespace(config):
    new_config = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            value = dict2namespace(value)
        setattr(new_config, key, value)
    return new_config


def parse_config(args):
    with open('config.yml', 'r') as f:
        config = yaml.load(f)
    return dict2namespace(config)


