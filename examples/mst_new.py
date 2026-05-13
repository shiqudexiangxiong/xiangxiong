import csv
import logging
import os
import time

from utils_examples import flatten_dict

from src.inference.dataset import Dataset
from src.inference.evaluation import Evaluator
from src.inference.pgm.inference import FactoredInference
from src.mechanisms.mst import MST
from src.mechanisms.utils_mechanisms import generate_all_kway_workload
from src.inference.evaluations.downstream import downstream_logistic
from src.inference.evaluations.covariance import covariance_error
from src.inference.evaluations.distances import (
    sw1_distance,
    tv_distance_final
)
from src.inference.evaluations.queries import threshold_query_error, counting_query_error

# 配置日志输出级别为 INFO，运行时会打印关键步骤信息
logging.basicConfig(level=logging.INFO)


# ─────────────────────────────────────────────
# 参数配置区：在此处直接修改实验参数，无需命令行传参
# ─────────────────────────────────────────────
params = {
    # 合成数据集的保存目录
    "savedir": "../data/datasets/acs_income/",

    # 训练数据集路径（离散化后的 CSV 文件）
    "train_dataset": "../data/datasets/acs_income/data_disc.csv",

    # 域描述文件路径（记录每列的取值范围/类别数，JSON 格式）
    "domain": "../data/datasets/acs_income/domain.json",

    # 差分隐私预算 ε：值越小隐私保护越强，但数据可用性越低
    "epsilon": 2.5,

    # 差分隐私参数 δ：允许以极小概率 δ 超出 ε 的隐私保证
    "delta": 0.00001,

    # PGM 推断引擎的学习率
    "lr": 1.0,

    # PGM 优化迭代次数
    "iters": 3000,

    # 模型允许的最大规模（控制联合分布的复杂度，防止内存溢出）
    "max_model_size": 1000,

    # 是否使用 warm start（复用上次优化结果作为初始值，可加速收敛）
    "warm_start": False,

    # 生成的合成记录条数；None 表示与原始数据集行数相同
    "records": None,
}


def run_mst_pgm(params: dict):
    """
    使用 MST（最大生成树）+ PGM（概率图模型）算法生成差分隐私合成数据。

    流程：
        1. 加载原始数据集与域描述
        2. 初始化 PGM 推断引擎（FactoredInference）
        3. 运行 MST 机制：在隐私预算内选取最优边集并估计联合分布
        4. 从学习到的分布中采样生成合成数据
        5. 评估合成数据质量（2-way 边际误差）
        6. 将结果追加写入 CSV 日志文件
    """

    # ── 步骤 1：加载数据集 ──────────────────────────────────────────────
    # Dataset.load 同时读取 CSV 数据和 JSON 域描述，
    # 域描述决定每列的离散取值数量，是后续图模型构建的基础
    data = Dataset.load(
        params["train_dataset"],
        params["domain"],
    )

    # ── 步骤 2：初始化 PGM 推断引擎 ────────────────────────────────────
    # FactoredInference 基于因子图进行边际推断，
    # 通过镜像下降（Mirror Descent）优化拟合噪声边际查询
    generation_engine = FactoredInference(
        domain=data.domain,
        hp=params,
    )

    # ── 步骤 3：初始化 MST 差分隐私机制 ────────────────────────────────
    # MST 在隐私预算内：
    #   (a) 用指数机制选取信息量最大的属性对（边）
    #   (b) 用高斯机制对选中边的 2-way 边际加噪
    # bounded=True 表示数据集大小本身也是公开的（bounded DP）
    mechanism = MST(
        epsilon=params["epsilon"],
        delta=params["delta"],
        bounded=True,
    )

    # ── 步骤 4：运行机制，生成合成数据 ─────────────────────────────────
    start_time = time.time()
    synth, loss = mechanism.run(
        data=data,
        engine=generation_engine,
    )
    end_time = time.time()

    print(f"总损失: {loss:.6f},  耗时: {end_time - start_time:.2f}s")

    # 若指定了保存目录，则将合成数据持久化为 CSV
    if params["savedir"]:
        os.makedirs(params["savedir"], exist_ok=True)
        synth.df.to_csv(
            os.path.join(params["savedir"], "mst_synth_data.csv"),
            index=False,
        )
        print(f"Synthetic data saved to {params['savedir']}mst_synth_data.csv")

    # ── 步骤 5：评估合成数据质量 ────────────────────────────────────────
    # 使用全量 2-way workload（所有属性对的联合边际）衡量误差
    # 注意：Evaluator 初始化时 synth 先传入原始 data 占位，
    #       随后通过 update_synth 替换为真正的合成数据
    print("Starting evaluation...")
    evaluator = Evaluator(
        data=data,
        synth=synth,
        workload=generate_all_kway_workload(data=data, degree=2),
    )
    evaluator.set_compression()   # 压缩稀疏边际以节省内存
    evaluator.update_synth(synth) # 替换为真实合成数据
    results, _ = evaluator.evaluate()
    paper_metrics = {}

    logging.info("  - Computing downstream task metrics...")
    paper_metrics.update(
        downstream_logistic(data.df, synth.df, label_col=data.df.columns[-1])
    )

    logging.info("  - Computing covariance error...")
    paper_metrics.update(covariance_error(data.df, synth.df))

    logging.info("  - Computing query errors...")
    paper_metrics.update(counting_query_error(data.df, synth.df))
    paper_metrics.update(threshold_query_error(data.df, synth.df))

    logging.info("  - Computing SW1 distance...")
    paper_metrics.update(sw1_distance(data.df, synth.df))

    logging.info("  - Computing TV distance...")
    paper_metrics.update(
        tv_distance_final(data.df, synth.df, domain_path=params["domain"])
    )
    # ── 步骤 6：汇总结果并写入 CSV 日志 ────────────────────────────────
    dataset_name = os.path.basename(os.path.dirname(params["train_dataset"]))

    experiment_results = {
        "dataset_name": dataset_name,
        "time": end_time - start_time,          # 实际运行耗时（秒）
        "loss": loss,
        # 展平超参数字典（排除路径类参数，保持日志简洁）
        **flatten_dict(
            {k: v for k, v in params.items()
             if k not in ["savedir", "train_dataset", "domain"]}
        ),
        # 展平评估指标字典（如各边际的最大误差、平均误差等）
        # **flatten_dict(results),
        **flatten_dict(paper_metrics)
    }

    results_file = os.path.join(params["savedir"], "mst_results.csv")
    file_exists = os.path.isfile(results_file)

    # 以追加模式写入，支持多次实验结果累积在同一文件中
    with open(results_file, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=experiment_results.keys())
        if not file_exists:
            writer.writeheader()   # 首次写入时添加表头
        writer.writerow(experiment_results)

    print(f"Results saved to {results_file}")


if __name__ == "__main__":
    # 直接调用函数，PyCharm 可通过右键 > Run 或 Shift+F10 执行
    for _ in range(4):
        run_mst_pgm(params)