import csv
import logging
import os
import time

from utils_examples import flatten_dict

from src.inference.dataset import Dataset
from src.inference.evaluation import Evaluator
from src.inference.pgm.inference import FactoredInference
from src.mechanisms.aim import AIM
from src.mechanisms.utils_mechanisms import generate_all_kway_workload
from src.inference.evaluations.downstream import downstream_logistic
from src.inference.evaluations.covariance import covariance_error
from src.inference.evaluations.distances import (
    sw1_distance,
    tv_distance_final
)
from src.inference.evaluations.queries import threshold_query_error, counting_query_error

# 配置日志输出级别为 INFO，运行过程中可在控制台看到进度信息
logging.basicConfig(level=logging.INFO)


# ─────────────────────────────────────────────
# 【参数配置区】在 PyCharm 中直接修改此处的值即可
# ─────────────────────────────────────────────
PARAMS = {
    # 合成数据集的保存目录
    "savedir": "../data/datasets/acs_income/",

    # 训练数据集路径（离散化后的 CSV 文件）
    "train_dataset": "../data/datasets/acs_income/data_disc.csv",

    # 域描述文件路径（JSON 格式，定义每列的取值范围/类别数）
    "domain": "../data/datasets/acs_income/domain.json",

    # 差分隐私预算 epsilon：值越小隐私保护越强，但数据质量越低
    "epsilon": 2.5,

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


def run_aim_pgm(params: dict):
    """
    使用 AIM + PGM 框架生成差分隐私合成数据，并评估数据质量。

    流程：
      1. 加载原始数据集及域描述
      2. 构建 PGM（概率图模型）推断引擎
      3. 构建 AIM 差分隐私机制
      4. 生成二阶边际工作负载（所有两列组合的统计量）
      5. 运行 AIM 机制，输出合成数据集
      6. 评估合成数据与原始数据之间的统计差异
      7. 将实验结果追加写入 CSV 文件
    """

    # ── Step 1：加载数据集 ──────────────────────────────────────────────────
    # Dataset.load 读取 CSV 并根据 domain.json 将列映射为离散整数编码
    data = Dataset.load(
        params["train_dataset"],
        params["domain"],
    )

    # ── Step 2：初始化 PGM 推断引擎 ────────────────────────────────────────
    # FactoredInference 基于因子图进行边际推断，用于将测量到的边际转换为联合分布
    generation_engine = FactoredInference(
        domain=data.domain,
        hp=params,
    )

    # ── Step 3：初始化 AIM 差分隐私机制 ────────────────────────────────────
    # AIM（Adaptive and Iterative Mechanism）自适应地选择最有信息量的边际进行测量
    # bounded=True 表示使用有界差分隐私（已知数据集大小）
    mechanism = AIM(
        epsilon=params["epsilon"],
        delta=params["delta"],
        max_model_size=params["max_model_size"],
        bounded=True,
    )

    # ── Step 4：生成工作负载 ────────────────────────────────────────────────
    # 工作负载为所有二阶（两列）边际的集合，用于评估合成数据与原始数据的统计相似度
    workload = generate_all_kway_workload(data=data, degree=2)

    # ── Step 5：运行 AIM 机制，生成合成数据 ────────────────────────────────
    start_time = time.time()
    synth, loss = mechanism.run(
        data=data,
        engine=generation_engine,
        workload=workload,
    )
    end_time = time.time()

    print(f"总损失: {loss:.4f}，耗时: {end_time - start_time:.2f} 秒")

    # 将合成数据保存为 CSV 文件
    if params["savedir"]:
        os.makedirs(params["savedir"], exist_ok=True)
        synth_path = os.path.join(params["savedir"], "aim_synth_data.csv")
        synth.df.to_csv(synth_path, index=False)
        print(f"合成数据已保存至: {synth_path}")

    # ── Step 6：评估合成数据质量 ────────────────────────────────────────────
    # Evaluator 计算合成数据与原始数据在所有二阶边际上的误差（如 L1/L2 距离）
    print("开始评估合成数据质量...")
    evaluator = Evaluator(
        data=data,
        synth=synth,
        workload=workload,
    )
    evaluator.set_compression()   # 压缩域以加速评估
    evaluator.update_synth(synth) # 更新评估器中的合成数据
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
    # ── Step 7：保存实验结果 ────────────────────────────────────────────────
    dataset_name = os.path.basename(os.path.dirname(params["train_dataset"]))

    # 将所有参数与评估指标合并为一行记录
    experiment_results = {
        "dataset_name": dataset_name,
        "time": time.time() - start_time,
        "loss": loss,
        # 过滤掉路径类参数，只保留超参数
        **flatten_dict(
            {
                k: v
                for k, v in params.items()
                if k not in ["savedir", "train_dataset", "domain"]
            }
        ),
        # **flatten_dict(results),
        **flatten_dict(paper_metrics)
    }

    # 以追加模式写入 CSV，首次运行时自动写入表头
    results_file = os.path.join(params["savedir"], "aim_results.csv")
    file_exists = os.path.isfile(results_file)

    with open(results_file, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=experiment_results.keys())
        if not file_exists:
            writer.writeheader()  # 仅首次创建文件时写入列名
        writer.writerow(experiment_results)

    print(f"实验结果已保存至: {results_file}")


# ─────────────────────────────────────────────
# 程序入口：在 PyCharm 中点击右键 → Run 即可执行
# ─────────────────────────────────────────────
if __name__ == "__main__":
    for _ in range(4):
        run_aim_pgm(PARAMS)