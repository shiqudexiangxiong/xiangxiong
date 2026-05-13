from examples.aim_new import run_aim_pgm
from examples.mst_new import run_mst_pgm

PARAMS = {
    # 合成数据集的保存目录
    "savedir": "../data/datasets/acs_income/",

    # 训练数据集路径（离散化后的 CSV 文件）
    "train_dataset": "../data/datasets/acs_income/data_disc.csv",

    # 域描述文件路径（JSON 格式，定义每列的取值范围/类别数）
    "domain": "../data/datasets/acs_income/domain.json",

    # 差分隐私预算 epsilon：值越小隐私保护越强，但数据质量越低
    "epsilon": 1.0,

    # 差分隐私参数 delta：通常设为 1/n^1.1（n 为数据集大小），控制隐私失败概率
    "delta": 0.00001,

    # PGM 推断引擎的学习率
    "lr": 1.0,

    # PGM 推断引擎的迭代次数
    "iters": 3000,

    # 模型允许的最大尺寸（控制联合分布图模型的复杂度，防止内存溢出）
    "max_model_size": 1000,

    # 是否使用 warm start（复用上一轮的模型参数以加速收敛）
    "warm_start": False,

    # 生成的合成数据记录数；None 表示与原始数据集等量
    "records": None,
}
if __name__ == "__main__":
    for _ in range(5):
        run_aim_pgm(params=PARAMS)
        run_mst_pgm(params=PARAMS)