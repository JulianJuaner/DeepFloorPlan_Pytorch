import yacs.config as config

def make_config(config_file):
    with open(config_file, 'r') as f:
        cfg = config.load_cfg(f)
    return cfg.clone()

if "__main__" in __name__:
    print(make_config('../../exps/01/exp.yaml'))