import os

def create_dirs(path,add_php=False):
    if not os.path.exists(path):
        os.makedirs(path)
    if add_php:
        cmd = "cp /eos/home-m/mmatthew/www/index/index.php %s"%path
        os.system(cmd)
