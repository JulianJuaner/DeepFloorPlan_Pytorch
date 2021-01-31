import yacs.config as config

def make_config(config_file):
    with open(config_file, 'r') as f:
        cfg = config.load_cfg(f)
    return cfg.clone()
    
# https://stackoverflow.com/questions/1305532/convert-nested-python-dict-to-object/1305663
class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


if "__main__" in __name__:
    print(make_config('../../exps/01/exp.yaml'))