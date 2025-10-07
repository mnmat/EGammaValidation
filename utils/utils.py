import os

def create_dirs(path,add_php=False):
    if not os.path.exists(path):
        os.makedirs(path)
    if add_php:
        cmd = "cp /eos/home-m/mmatthew/www/index/index.php %s"%path
        os.system(cmd)

def get_training_test_files(path,training_dir="s4Flat_genMatched",test_dir="s5Reg_genMatched"):
    training_path = os.path.join(path,training_dir)
    test_path = os.path.join(path,test_dir)

    training_file = "HLTAnalyzerTree_IDEAL_Flat_train.root"
    test_file = "Run3HLT_IdealIC_IdealTraining_stdVar_stdCuts_ntrees1500_applied.root"

    return os.path.join(training_path,training_file), os.path.join(test_path,test_file)

def modify_tree(df):
    df["old_regressedEnergy"] = df["regressedEnergy"]
    df["regressedEnergy"] = df["rawEnergy"]*df["regEEMean"]
    
    df["frac_rawEnergy_genEnergy"] = df["rawEnergy"]/df["eg_gen_energy"]
    df["frac_old_regEnergy_genEnergy"] = df["old_regressedEnergy"]/df["eg_gen_energy"]
    df["frac_regEnergy_genEnergy"] = df["regressedEnergy"]/df["eg_gen_energy"]
    
    df["old_regEEMean"] = df["old_regressedEnergy"]/df["rawEnergy"]
    return df