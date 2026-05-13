import os

from examples.privpgd_new import run_privpgd

if __name__ == "__main__":
    # for ws in ["uniform", "entropy","importance"]:
    # #     # uniform: Original equal allocation (baseline)ε分配方式
    # #     # variance: Allocates more budget to high-variance marginals
    # #     # entropy: Allocates more budget to high-entropy marginals
    # #     # importance: Combines variance and entropy (50-50 mix)
    # #     # custom: Allows manual weight specification
    #     for para in ["renyi", "gp"]:#噪声机制添加方式
    # #     # gaussian:original method
    # #     # renyi:RDP method
    # #     # gp:gaussian process DP
    #         for _ in range(10):
    #             run_privpgd(weight_strategy=ws,noise_type=para)
    savedir ="../data/datasets/acs_income/"
    train_dataset=os.path.join(savedir, "data_disc.csv")
    domain=os.path.join(savedir, "domain.json")
    # for _ in range(5):
    #     run_privpgd(epsilon=1.0,savedir=savedir,train_dataset=train_dataset,domain=domain,weight_strategy="uniform",noise_type="gp")
    # for _ in range(5):
    #     run_privpgd(epsilon=2.5, savedir=savedir, train_dataset=train_dataset, domain=domain, weight_strategy="uniform",
    #                 noise_type="gp")
    run_privpgd(epsilon=2.5, savedir=savedir, train_dataset=train_dataset, domain=domain, weight_strategy="uniform",noise_type="gp")