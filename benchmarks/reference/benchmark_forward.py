"""CUDA Event benchmark and PyTorch Profiler harness for P1 reference paths.

The sweep compares Dense SDPA, masked-dense GQA, and gathered sparse GQA
on fixed prefix selections. Masked/gathered output and LSE parity is checked
before timing; Dense SDPA is a full-history engineering baseline, not a
same-support correctness oracle.

Timing covers the complete callable. Profiler output explains execution
structure and is not used as a latency estimate. The default two-sample
configuration preserves the historical wiring check and is not statistically
stable.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable
from pathlib import Path
from typing import TypedDict

import torch

from msa_lab.backends.base import AttentionOutput
from msa_lab.backends.dense_sdpa import DenseSDPABackend
from msa_lab.contracts.support import (
    block_ids_to_group_mask,
    validate_fixed_block_ids,
)
from msa_lab.reference.gathered_sparse import gathered_sparse_gqa
from msa_lab.reference.masked_dense import masked_dense_gqa


# region C05 配置

# C05 冒烟配置
B = 1
N = 128
HQ = 8
HKV = 2
D = 64
BLOCK_SIZE = 32
KSEL = 2

# 正整数向上整除：计算覆盖 N 个词元所需的键块数量。
# KSEL 只是固定冒烟值；扫描合法性必须使用 N_BLOCKS，否则会因为冒烟值是 2
# 而错误拒绝 ksel=3/4。
N_BLOCKS = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
KSEL_SWEEP = (1, 2, 3, 4)
DTYPE = torch.float32
DEVICE = torch.device("cuda")
WARMUP = 2
REPETITIONS = 2

# profiler 固定只分析一个代表 Ksel，不重复执行整个 Ksel 扫描。
PROFILE_KSEL = 2

# 这些预热在创建 profiler 之前执行，完全不会被采集。
PROFILE_WARMUP = 2

# profiler 内部 schedule 的一个 cycle：WAIT 不记录，WARMUP 让 profiler
# 自身热身，ACTIVE 才保存事件。三个阶段都会真实调用 forward。
PROFILE_WAIT_STEPS = 1
PROFILE_SCHEDULE_WARMUP_STEPS = 1
PROFILE_ACTIVE_STEPS = 1

# repeat 表示重复多少个完整的 WAIT/WARMUP/ACTIVE cycle。
PROFILE_REPEAT = 1

# 每张文本表最多显示的行数。
PROFILE_ROW_LIMIT = 20

# 按调用栈聚合时，把最近多少层栈作为分组键。
PROFILE_STACK_DEPTH = 5

# Chrome traces are written under the CLI output directory and are ignored by Git.

# endregion


# region C05 顶层结果合同


class KForwardSweepRow(TypedDict):
    """描述一个 Ksel 实验点的完整前向证据。"""

    support: dict[str, int | float]
    samples_ms: dict[str, list[float]]
    summary_ms: dict[str, dict[str, int | float]]


class ForwardProfileTables(TypedDict):
    """Store aggregate profiler outputs for one forward callable."""

    # 下面五张表都来自同一份 ACTIVE 事件，只是分组和排序方式不同。
    cuda_table: str
    cpu_table: str
    memory_table: str
    flops_table: str
    stack_table: str

    # Chrome trace JSON 的绝对路径，用于查看 CPU/CUDA 时间线。
    trace_path: str


# endregion


# Input fixture and metadata contract


def make_smoke_fixture() -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """构造 Q/K/V 和合法固定选择元数据，用于冒烟运行。"""
    # Create deterministic CUDA smoke-test inputs.

    # 在函数内部重新设种子，使重复调用产生相同的 Q/K/V；manual_seed 会重置
    # PyTorch 全局随机数生成器状态，而不是为本函数创建私有生成器。
    torch.manual_seed(0)

    # 从标准正态分布生成随机张量。
    q = torch.randn(
        B, N, HQ, D,
        device=DEVICE,
        dtype=DTYPE,
    )

    k = torch.randn(
        B, N, HKV, D,
        device=DEVICE,
        dtype=DTYPE,
    )

    v = torch.randn(
        B, N, HKV, D,
        device=DEVICE,
        dtype=DTYPE,
    )

    # 每个元数据行固定选择 [0, 1]。即使 block 1 对早期 query 尚在未来，
    # block 0 仍提供非空因果支持集，因此该原始选择合法。
    block_ids_row = torch.tensor([0, 1], device=DEVICE, dtype=torch.long)

    # reshape 显式插入大小为 1 的 B/N/Hkv 轴：[Ksel] -> [1,1,1,Ksel]。
    # expand 将它广播成 [B,N,Hkv,Ksel] 的零步长视图，而不是复制每一行。
    # 冻结版可调用对象只读取元数据，因此不需要 clone。
    block_ids = block_ids_row.reshape(1, 1, 1, KSEL).expand(B, N, HKV, KSEL)

    return q, k, v, block_ids


def validate_smoke_fixture(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_ids: torch.Tensor,
) -> None:
    """在调用任何注意力实现之前验证冒烟测试输入。"""
    # Validate the fixture before calling any implementation.
    # 1. C01 元数据合同
    validate_fixed_block_ids(
        block_ids=block_ids,
        seq_len=N,
        block_size=BLOCK_SIZE,
    )

    # 2. GQA 配置合同
    assert HQ % HKV == 0

    # 3. Q/K/V 形状
    assert q.shape == (B, N, HQ, D)
    assert k.shape == (B, N, HKV, D)
    assert v.shape == (B, N, HKV, D)


    # 4. block_ids 形状
    assert block_ids.shape == (B, N, HKV, KSEL)

    # 5. 数据类型
    assert q.dtype == DTYPE
    assert k.dtype == DTYPE
    assert v.dtype == DTYPE
    assert block_ids.dtype == torch.long

    # 6. 设备
    # DEVICE=torch.device("cuda") 是“当前 CUDA 设备”的未解析别名；实际分配后
    # tensor.device 会带具体逻辑编号（例如 cuda:0），两者不做严格对象相等。
    # 比较 .type 用来确认加速器类型；它本身不证明多个张量位于同一张具体 GPU。
    # 当前测试输入全部由同一个 DEVICE 创建；若以后接收外部测试输入，还应额外
    # 检查 q.device == k.device == v.device == block_ids.device。
    assert q.device.type  == DEVICE.type
    assert k.device.type == DEVICE.type
    assert v.device.type == DEVICE.type
    assert block_ids.device.type == DEVICE.type

    return None


# endregion


# Forward smoke checks and same-support parity


def assert_tensor_contract(
    tensor: torch.Tensor,
    expected_shape: tuple[int, ...],
) -> None:
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == expected_shape
    assert tensor.dtype == DTYPE
    # 输出张量报告具体设备（如 cuda:0），DEVICE 当前则是未指定编号的 cuda
    # 别名，所以这里只比较加速器类型，避免 cuda:0 != cuda
    # 造成假失败。
    assert tensor.device.type == DEVICE.type
    # 有限值检查会把 CUDA 布尔标量取回 Python 并触发同步；冒烟验证可以接受，
    # 但后续不能把它放进 CUDA 事件的被计时区间。
    assert torch.isfinite(tensor).all().item()


def run_smoke_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_ids: torch.Tensor,
) -> None:
    dense_backend = DenseSDPABackend()

    # DenseSDPABackend 不接收 block_ids：它计算完整稠密支持集。causal=True
    # 根据分数张量的 query/key 序列轴下标应用下三角掩码，使 query i 只见
    # key j<=i；causal=False 则允许完整双向注意力，包括自回归语义中的未来 token。
    dense_backend_result = dense_backend(
        q=q,
        k=k,
        v=v,
        causal=True,
        return_lse=False,
        return_aux=False
    )

    # 掩码稠密与聚集稀疏都接收 block_ids，因为二者表达相同的选中支持集注意力；
    # 前者保留 N×N 分数张量作为语义标准。
    masked_dense_gqa_result = masked_dense_gqa(
        q=q,
        k=k,
        v=v,
        block_ids=block_ids,
        block_size=BLOCK_SIZE
    )

    # 聚集参考实现只对有效支持集中的 M 个 K/V token 计算注意力。
    gathered_sparse_gqa_result = gathered_sparse_gqa(
        q=q,
        k=k,
        v=v,
        block_ids=block_ids,
        block_size=BLOCK_SIZE
    )

    expected_output_shape = (B, N, HQ, D)
    expected_lse_shape = (B, N, HQ)

    # AttentionOutput 是数据类容器，形状、数据类型和设备属于其中的张量字段，
    # 而不是容器本身。稠密实现本轮 return_lse=False，所以 lse 应为 None；
    # 两个稀疏参考实现则返回 output 与自然对数 LSE。
    assert isinstance(dense_backend_result, AttentionOutput)
    assert_tensor_contract(
        dense_backend_result.output,
        expected_output_shape
    )
    assert dense_backend_result.lse is None

    assert isinstance(masked_dense_gqa_result, AttentionOutput)
    assert_tensor_contract(
        masked_dense_gqa_result.output,
        expected_output_shape,
    )
    assert masked_dense_gqa_result.lse is not None
    assert_tensor_contract(
        masked_dense_gqa_result.lse,
        expected_lse_shape,
    )

    assert isinstance(gathered_sparse_gqa_result, AttentionOutput)
    assert_tensor_contract(
        gathered_sparse_gqa_result.output,
        expected_output_shape,
    )
    assert gathered_sparse_gqa_result.lse is not None
    assert_tensor_contract(
        gathered_sparse_gqa_result.lse,
        expected_lse_shape,
    )

    # 正确性配对只包含掩码稠密与聚集实现：两者支持集相同，只是执行路径不同。
    # assert_close 允许不同矩阵乘法顺序产生的正常浮点舍入误差。稠密实现使用
    # 完整因果支持集，不能与只看 blocks [0,1] 的稀疏结果比较。
    torch.testing.assert_close(
        masked_dense_gqa_result.output,
        gathered_sparse_gqa_result.output,
    )

    torch.testing.assert_close(
        masked_dense_gqa_result.lse,
        gathered_sparse_gqa_result.lse,
    )


# endregion


# CUDA Event timing


def measure_cuda_ms(
    fn: Callable[[], AttentionOutput],
    *,
    warmup: int,
    repetitions: int,
) -> list[float]:
    """重复测量一个零参数 CUDA 前向可调用对象。

    CUDA 事件的基础身份是流进度标记：无论是否启用计时，record 后它都会经历
    等待完成到已经完成的状态变化，可用于查询、同步和依赖。enable_timing=True
    只是在 GPU 执行到标记时额外保存设备时间戳，从而允许两个已记录事件通过
    start.elapsed_time(end) 求相对毫秒差。

    这是设备事件指标，不是唯一计时方法。另一种方法是在 CPU perf_counter 前后
    各放置 CUDA 同步，从而测量主机调用者观察到的端到端延迟。两种指标回答不同
    问题，不能在不说明的情况下相互替换。
    """
    # Timing boundary:
    # - fn 已经通过闭包绑定 Q/K/V 和后端专用参数；
    # - 测试输入创建与正确性断言必须留在本函数之外；
    # - 计时器只负责预热、同步、CUDA 事件和原始毫秒样本。
    #

    samples_ms: list[float] = []

    if warmup < 0:
        raise ValueError("warmup must be non-negative")

    if repetitions <= 0:
        raise ValueError("repetitions must be positive")

    # 预热仍会真正调用 fn，但不创建事件、不保存样本。
    # 因此 fn 的总调用次数是 warmup + repetitions。
    for _ in range(warmup):
        fn()

    # CUDA 启动默认异步；正式计时前等待当前 GPU 所有流上已经提交的预热工作，
    # 建立不计入任何样本的干净边界。
    torch.cuda.synchronize()

    # 每次重复测量使用独立的一对事件。
    for _ in range(repetitions):

        # 构造 Event 只创建尚未绑定流位置的记录器；enable_timing=True 表示它
        # 未来被 record 且 GPU 执行到标记时，要额外保存设备时间戳。
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        # record() 把标记排入当前 CUDA 流，但不会立刻把事件标记为已经完成。
        # GPU 未来执行到标记时，事件才完成并保存时间戳。fn 把被测操作排在
        # start 后，end 再排在这些操作之后。这个推理要求工作位于当前流，
        # 或其他流已经建立显式依赖；当前冻结参考实现不进入多流生产级调度。
        start.record()
        fn()
        end.record()

        # 同一流使用先进先出顺序；end.synchronize() 不负责创造这个顺序，
        # 只阻塞 CPU，直到 GPU 执行到 end。此时 start、fn、end 均已完成，
        # 但无需等待 end 之后的工作，也无需每轮同步整张 GPU。
        end.synchronize()

        # elapsed_time 返回 end 时间戳减 start 时间戳的相对毫秒差，
        # 不是查询某个事件的绝对设备时钟。
        elapsed_ms = start.elapsed_time(end)
        samples_ms.append(elapsed_ms)

    # 每次重复测量产生一个浮点毫秒值；暂时保留原始样本，不在计时器内计算
    # 均值、中位数或分位数，也不混入正确性断言。
    return samples_ms


def run_forward_timing(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_ids: torch.Tensor,
    *,
    warmup: int,
    repetitions: int,
) -> dict[str, list[float]]:
    """统一三种接口，以相同配置计时，并返回具名原始数据。

    返回字典可以把测量与展示分开。main 可以打印数据，后续层可以汇总数据；
    如果此处返回 None 且函数内部不打印，就会静默丢弃已经采集的样本。
    """
    # 后端对象在闭包外构造：重复测量应该测前向执行，而不是反复构造 Python 对象。
    dense_backend = DenseSDPABackend()

    # 闭包捕获现有测试输入的引用，不复制或重新创建 Q/K/V。传给计时器的是
    # dense_forward 函数本身，而不是 dense_forward()；后者会在 CUDA 事件记录前执行。
    def dense_forward() -> AttentionOutput:
        # 稠密基线使用完整因果支持集，不接收 block_ids。
        return dense_backend(
            q=q,
            k=k,
            v=v,
            causal=True,
            return_lse=False,
            return_aux=False
        )

    def masked_forward() -> AttentionOutput:
        # 掩码稠密计时包含其冻结的支持集展开与校验成本。
        return masked_dense_gqa(
            q=q,
            k=k,
            v=v,
            block_ids=block_ids,
            block_size=BLOCK_SIZE
        )

    def gathered_forward() -> AttentionOutput:
        # 聚集实现计时刻意包含其冻结的 Python 循环、支持集展开、间接聚集、
        # 小矩阵乘法和内核启动间隔。
        return gathered_sparse_gqa(
            q=q,
            k=k,
            v=v,
            block_ids=block_ids,
            block_size=BLOCK_SIZE
        )

    # 三个可调用对象使用相同测试输入、预热次数和重复次数，但各自独立预热。
    # 这里测的是完整冻结可调用对象的实际事件区间，不是只按 FLOPs 推导的理想
    # 内核时间；参考实现延迟高不否定稀疏数学计算量更小。
    dense_samples = measure_cuda_ms(
        fn=dense_forward,
        warmup=warmup,
        repetitions=repetitions
    )

    masked_samples = measure_cuda_ms(
        fn=masked_forward,
        warmup=warmup,
        repetitions=repetitions
    )

    gathered_samples = measure_cuda_ms(
        fn=gathered_forward,
        warmup=warmup,
        repetitions=repetitions
    )

    return {
        "dense_sdpa": dense_samples,
        "masked_dense_gqa": masked_samples,
        "gathered_sparse_gqa": gathered_samples,
    }


# endregion


# Timing sample summaries

def summarize_samples(
    samples_ms: list[float],
) -> dict[str, int | float]:
    """Summarize CUDA Event samples."""
    if not samples_ms:
        raise ValueError("samples_ms must be non-empty")

    count = len(samples_ms)
    median_ms = statistics.median(samples_ms)
    min_ms = min(samples_ms)
    max_ms = max(samples_ms)


    return {
        "count": count,
        "median_ms": median_ms,
        "min_ms": min_ms,
        "max_ms": max_ms
    }


# endregion


# Ksel sweep and evidence ledger


def make_prefix_block_ids(
    *,
    ksel: int,
) -> torch.Tensor:
    """为受控 Ksel 扫描构造一份合法前缀选择张量。

    每个 ``[b, q, hkv]`` 行必须包含唯一 block ID ``[0, ..., ksel - 1]``，
    形状为 ``[B, N, HKV, ksel]``。即使后续选中 block 对早期 query 仍位于
    未来，block 0 也能保证早期因果行非空。

    前缀选择是受控合成扫描，不是学习得到的 Top-k：当 N=128、block_size=32、
    ksel=3 时，选择 blocks [0,1,2]，分别代表 token 范围 [0:32]、[32:64]
    和 [64:96]。因果规则仍会移除未来 token，因此有效 M 随行变化，不会恒等于
    ksel * block_size。
    """
    # Reject selections outside the available prefix blocks.
    if not (1 <= ksel <= N_BLOCKS):
        raise ValueError(f"ksel must be between 1 and {N_BLOCKS}, but got {ksel}")

    # Block IDs are integer metadata, not floating-point Q/K/V values.
    prefix_row = torch.arange(ksel, device=DEVICE, dtype=torch.long)

    # reshape 让轴含义显式化，expand 随后创建跨 B/N/Hkv 共享的只读广播视图；
    # 它不会移动设备或改变数据类型。
    prefix_blocks_id = prefix_row.reshape(1, 1, 1, ksel).expand(B, N, HKV, ksel)

    # Validate metadata before returning it to the benchmark.
    validate_fixed_block_ids(
        block_ids=prefix_blocks_id,
        seq_len=N,
        block_size=BLOCK_SIZE,
    )

    return prefix_blocks_id


def summarize_effective_support(
    block_ids: torch.Tensor,
) -> dict[str, int | float]:
    """汇总一份选择张量产生的精确因果 token 支持集。"""
    # Expand selected blocks into the effective causal token support.
    # [B, N, Hkv, N]
    group_mask = block_ids_to_group_mask(
        block_ids=block_ids,
        seq_len=N,
        block_size=BLOCK_SIZE,
    )
    # Count supported key tokens for every query/KV-head row.
    group_mask_row_count = group_mask.sum(dim=-1)

    # Convert summary values to Python scalars outside the timed region.
    # 此台账位于所有 CUDA 事件计时区间之外，所以这里允许 .item() 触发同步。
    min_M = group_mask_row_count.min().item()
    max_M = group_mask_row_count.max().item()
    mean_M = group_mask_row_count.float().mean().item()

    return {
        "min_M": min_M,
        "max_M": max_M,
        "mean_M": mean_M,
    }


def run_k_forward_sweep(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    ksel_values: tuple[int, ...],
    warmup: int,
    repetitions: int,
) -> dict[int, KForwardSweepRow]:
    """按 Ksel 执行正确性、工作量和延迟的完整前向扫描。"""
    sweep_results: dict[int, KForwardSweepRow] = {}

    for ksel in ksel_values:
        # 一条实验记录的完整生命周期：构造控制变量，先通过正确性
        # 门禁，再记录实际工作量与延迟证据。所有断言和 .item() 同步都在
        # CUDA 事件计时区间之外。
        block_ids = make_prefix_block_ids(ksel=ksel)
        run_smoke_forward(
            q=q,
            k=k,
            v=v,
            block_ids=block_ids,
        )

        support_summary = summarize_effective_support(block_ids=block_ids)
        samples_ms = run_forward_timing(
            q=q,
            k=k,
            v=v,
            block_ids=block_ids,
            warmup=warmup,
            repetitions=repetitions,
        )
        summary_ms = {
            backend_name: summarize_samples(backend_samples)
            for backend_name, backend_samples in samples_ms.items()
        }

        sweep_results[ksel] = {
            "support": support_summary,
            "samples_ms": samples_ms,
            "summary_ms": summary_ms,
        }

    return sweep_results


# endregion


# CPU/CUDA Profiler attribution


def profile_cuda_callable(
    fn: Callable[[], AttentionOutput],
    *,
    label: str,
    warmup: int,
    wait_steps: int,
    schedule_warmup_steps: int,
    active_steps: int,
    repeat: int,
    row_limit: int,
    stack_depth: int,
    trace_dir: str,
) -> ForwardProfileTables:
    """对一个零参数 CUDA forward 生成多维 profiler 归因证据。

    函数按以下顺序工作：
    1. Validate the profiling configuration;
    2. 在 profiler 外预热并同步；
    3. 准备 trace 目录和 callback 共享结果容器；
    4. 构造 WAIT/WARMUP/ACTIVE schedule；
    5. 在 profiler 内逐 step 调用 fn；
    6. ACTIVE cycle 结束时由 callback 生成表格和 Chrome trace；
    7. 返回 callback 写入的证据。

    profiler 用于回答成本花在哪里，不用于重新估计稳定延迟。
    正式延迟仍以 measure_cuda_ms 的 CUDA Event 样本为准。
    """
    # 0. 参数合同。这些检查全部发生在 profiler 启动之前。
    if not label:
        raise ValueError("label must be non-empty")

    if warmup < 0:
        raise ValueError("warmup must be non-negative")

    if wait_steps < 0:
        raise ValueError("wait_steps must be non-negative")

    if schedule_warmup_steps < 0:
        raise ValueError("schedule_warmup_steps must be non-negative")

    if active_steps <= 0:
        raise ValueError("active_steps must be positive")

    if repeat <= 0:
        raise ValueError("repeat must be positive")

    if row_limit <= 0:
        raise ValueError("row_limit must be positive")

    if stack_depth <= 0:
        raise ValueError("stack_depth must be positive")

    if not trace_dir:
        raise ValueError("trace_dir must be non-empty")

    # 1. profiler 外普通预热。fn 会真实执行 warmup 次，但此时还没有
    # 创建 torch.profiler.profile，所以这些调用不会出现在任何表格或 trace 中。
    for _ in range(warmup):
        fn()

    # fn() 只保证工作已提交给 CUDA。这里等待外部预热的设备工作
    # 全部完成，防止它们和后面的 profiler schedule 在 GPU 上重叠。
    torch.cuda.synchronize()

    # 2. 准备 Chrome trace 输出目录。Path() 只构造路径对象，不会创建
    # 目录；mkdir() 才执行文件系统操作。parents=True 表示递归创建缺失的
    # 父目录，exist_ok=True 表示目录已存在时不报错。创建目录必须位于
    # profiler 之外，否则会把 harness 的文件系统开销记入 CPU trace。
    trace_dir_path = Path(trace_dir)
    trace_dir_path.mkdir(parents=True, exist_ok=True)

    # on_trace_ready callback 的返回值不会自动变成外层函数的返回值。
    # 因此先在外层创建一个可变字典；callback 稍后把表格和路径
    # 写入它，profiler 结束后外层函数再读取它。key 和 value 都是字符串。
    captured: dict[str, str] = {}

    # 此处只定义 callback，不会立即执行。后面将 trace_handler 传给
    # on_trace_ready；每个 ACTIVE cycle 完成时，profiler 才会调用它，并把
    # 当前 profiler 对象作为 prof 参数传入。
    def trace_handler(prof: torch.profiler.profile) -> None:
        # key_averages() 将大量原始事件按键聚合成算子统计。第一份视图
        # 把输入 shape 也纳入键：同名算子如果输入 shape 不同，会分成不同行。
        # 这要求后面 profile(..., record_shapes=True) 已经采集 shape。
        averages_by_shape = prof.key_averages(
            group_by_input_shape=True,
        )

        # 第二份视图把最近 stack_depth 层调用栈纳入键：同一算子
        # 如果由不同源码路径调用，可以分成不同行。这要求后面
        # profile(..., with_stack=True) 已经采集调用栈。
        averages_by_stack = prof.key_averages(
            group_by_stack_n=stack_depth,
        )

        # schedule 会自动为每个逻辑 step 增加 ProfilerStep* 顶层范围。它在
        # Chrome trace 中有助于识别 step 边界，但其 CUDA 时间与内部算子/kernel
        # 重叠，不应参与算子热点排名。这里只从两份聚合视图中删除它；
        # 原始 prof 事件和随后导出的 Chrome trace 保持不变。
        averages_by_shape[:] = [
            event for event in averages_by_shape
            if event.key != "ProfilerStep*"
        ]
        averages_by_stack[:] = [
            event for event in averages_by_stack
            if event.key != "ProfilerStep*"
        ]

        # table() 只对已聚合的事件排序并渲染文本，不会重新执行 fn。
        # CUDA 表回答“哪些算子/kernel 的自身设备时间最高”。
        captured["cuda_table"] = averages_by_shape.table(
            sort_by="self_cuda_time_total",
            row_limit=row_limit,
            header=f"{label} — 按自身 CUDA 时间排序",
        )

        # CPU 表回答“哪些算子/调度活动的自身主机时间最高”。
        captured["cpu_table"] = averages_by_shape.table(
            sort_by="self_cpu_time_total",
            row_limit=row_limit,
            header=f"{label} — 按自身 CPU 时间排序",
        )

        # profile_memory=True 后可以按算子自身引起的 CUDA 张量内存变化排序。
        # 这不是 nvidia-smi 报告的整卡显存占用，负值通常表示释放或归还。
        captured["memory_table"] = averages_by_shape.table(
            sort_by="self_cuda_memory_usage",
            row_limit=row_limit,
            header=f"{label} — 按自身 CUDA 内存变化排序",
        )

        # with_flops=True 只会为 profiler 已实现公式的算子估算 FLOPs。
        # 某行的 FLOPs 为空不表示该操作没有数学工作，只表示 profiler
        # 没有为它提供估算。
        captured["flops_table"] = averages_by_shape.table(
            sort_by="flops",
            row_limit=row_limit,
            header=f"{label} — 按 profiler FLOPs 估算排序",
        )

        # 调用栈视图按 Self CPU 排序，用于把 host-side 热点连回源码路径。
        captured["stack_table"] = averages_by_stack.table(
            sort_by="self_cpu_time_total",
            row_limit=row_limit,
            header=f"{label} — 按调用栈分组的 CPU 热点",
        )

        # Chrome trace 保留时间线，而不是 key_averages() 的聚合行。.resolve()
        # 将相对路径转成绝对路径；step_num 进入文件名，避免多个 cycle
        # 的 JSON 相互覆盖。export_chrome_trace() 真正写入文件。
        trace_path = (
            trace_dir_path / f"{label}_step{prof.step_num}.json"
        ).resolve()
        prof.export_chrome_trace(str(trace_path))

        # captured 的 value 合同是 str，因此 Path 要转成字符串后保存。
        captured["trace_path"] = str(trace_path)

    # 3. 构造 schedule 状态机。schedule 本身只根据 step 编号返回当前
    # profiler 动作，不会自动调用 fn。默认配置的一个 cycle 是：
    #
    #   step 0: WAIT   — fn 真执行，profiler 不采集
    #   step 1: WARMUP — fn 真执行，profiler 准备采集，但不保留事件
    #   step 2: ACTIVE — fn 真执行，profiler 保留事件
    #
    # repeat 决定这个 cycle 重复多少次。
    profile_schedule = torch.profiler.schedule(
        wait=wait_steps,
        warmup=schedule_warmup_steps,
        active=active_steps,
        repeat=repeat,
    )
    total_schedule_steps = (
        wait_steps + schedule_warmup_steps + active_steps
    ) * repeat

    # 4. 启动 profiler 并绑定数据采集策略。下面的参数分为四组：
    # - activities：采集主机端 PyTorch 活动和设备端 CUDA 活动；
    # - schedule/on_trace_ready：决定什么时候采集，以及 cycle 完成后如何消费证据；
    # - record_shapes/profile_memory/with_stack/with_flops：决定额外采集哪些维度；
    # - acc_events：决定 repeat>1 时 key_averages() 是否累积前面 cycle 的事件。
    #
    # 这些附加数据都会增加 profiler 开销，所以表中的绝对时间不能
    # 替代 CUDA Event benchmark。
    with torch.profiler.profile(
        # 同时采集 CPU 算子/调度活动和 CUDA kernel/拷贝活动。
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],

        # profiler 每次 prof.step() 后使用 profile_schedule 决定下一个状态。
        schedule=profile_schedule,

        # 一个 ACTIVE cycle 完成时，profiler 自动调用 trace_handler(prof)。
        on_trace_ready=trace_handler,

        # 为原始事件保留算子输入 shape，供 group_by_input_shape=True 聚合。
        record_shapes=True,

        # 记录算子引起的张量内存分配与释放变化。
        profile_memory=True,

        # 保留调用栈，供 group_by_stack_n 聚合并连回源码路径。
        with_stack=True,

        # 对 profiler 已支持的算子估算 FLOPs。
        with_flops=True,

        # repeat>1 时保留并累积前面 cycle 的 FunctionEvents。当前 repeat=1，
        # 所以最终表格仍只表示一个 ACTIVE cycle。
        acc_events=True,
    ) as prof:
        # 进入 with 块后 profiler 开始按 schedule 跟踪 step。每个循环迭代
        # 对应一个逻辑 step：先在当前状态下执行 fn，再调用 prof.step()
        # 结束该 step 并让状态机前进。
        #
        # 不用 record_function 包住整次 forward：这种顶层范围与子 kernel
        # 的 CUDA 时间重叠，会扭曲 Self CUDA 聚合表。backend 身份已由
        # 独立 trace 文件名和表头记录。
        for _ in range(total_schedule_steps):
            fn()
            prof.step()

    # 5. 退出 with 块时，profiler 完成收尾。只要 schedule 曾完成一个
    # ACTIVE cycle，trace_handler 就应该已将证据写入 captured。空字典说明
    # schedule 和循环 step 数没有真正到达一次可保存的 ACTIVE 结束点。
    if not captured:
        raise RuntimeError(
            "profiler produced no active trace; check schedule/step configuration"
        )

    # 6. 返回类型合同要求的六份证据。这里显式列出 key，而不是直接
    # return captured，可以在返回边界及时暴露 callback 遗漏任何必需字段的问题。
    return {
        "cuda_table": captured["cuda_table"],
        "cpu_table": captured["cpu_table"],
        "memory_table": captured["memory_table"],
        "flops_table": captured["flops_table"],
        "stack_table": captured["stack_table"],
        "trace_path": captured["trace_path"],
    }


def run_forward_profiling(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_ids: torch.Tensor,
    *,
    warmup: int,
    wait_steps: int,
    schedule_warmup_steps: int,
    active_steps: int,
    repeat: int,
    row_limit: int,
    stack_depth: int,
    trace_dir: str,
) -> dict[str, ForwardProfileTables]:
    """将三种不同前向接口适配成闭包，再分别产生 profiler 证据。

    这一层只解决“要 profile 哪些 backend”和“如何绑定各自入参”。
    schedule、事件聚合和 trace 导出全部交给 profile_cuda_callable。
    """
    # backend Python 对象在 profiler 会话外构造，不属于任何 forward 证据。
    dense_backend = DenseSDPABackend()

    # 三个闭包都是零参数 Callable[[], AttentionOutput]。它们不拷贝
    # Q/K/V/block_ids，只捕获已存在张量的引用并补齐 backend 专用参数。
    # fixture 创建、元数据验证和数值正确性断言仍属于 main，不进入
    # 这些闭包，因此也不会进入 profiler ACTIVE 证据。
    def dense_forward() -> AttentionOutput:
        return dense_backend(
            q=q,
            k=k,
            v=v,
            causal=True,
            return_lse=False,
            return_aux=False,
        )

    def masked_forward() -> AttentionOutput:
        return masked_dense_gqa(
            q=q,
            k=k,
            v=v,
            block_ids=block_ids,
            block_size=BLOCK_SIZE,
        )

    def gathered_forward() -> AttentionOutput:
        return gathered_sparse_gqa(
            q=q,
            k=k,
            v=v,
            block_ids=block_ids,
            block_size=BLOCK_SIZE,
        )

    # 字典的三个 value 会按 Python 表达式求值顺序依次调用
    # profile_cuda_callable，所以会产生三个完全独立的 profiler session。
    # 这使每张表的算子调用次数和时间只属于一个 backend。所有
    # profiler 参数都原样转发，保证三种 backend 使用相同采集策略。
    return {
        "dense_sdpa": profile_cuda_callable(
            fn=dense_forward,
            label="dense_sdpa_forward",
            warmup=warmup,
            row_limit=row_limit,
            wait_steps=wait_steps,
            schedule_warmup_steps=schedule_warmup_steps,
            active_steps=active_steps,
            repeat=repeat,
            stack_depth=stack_depth,
            trace_dir=trace_dir,
        ),
        "masked_dense_gqa": profile_cuda_callable(
            fn=masked_forward,
            label="masked_dense_gqa_forward",
            warmup=warmup,
            row_limit=row_limit,
            wait_steps=wait_steps,
            schedule_warmup_steps=schedule_warmup_steps,
            active_steps=active_steps,
            repeat=repeat,
            stack_depth=stack_depth,
            trace_dir=trace_dir,
        ),
        "gathered_sparse_gqa": profile_cuda_callable(
            fn=gathered_forward,
            label="gathered_sparse_gqa_forward",
            warmup=warmup,
            row_limit=row_limit,
            wait_steps=wait_steps,
            schedule_warmup_steps=schedule_warmup_steps,
            active_steps=active_steps,
            repeat=repeat,
            stack_depth=stack_depth,
            trace_dir=trace_dir,
        ),
    }


# endregion


# CLI entry point


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark the P1 transparent attention references on a CUDA GPU."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/p1-c05"),
        help="Directory for JSON results, profiler tables, and Chrome traces.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=WARMUP,
        help="Warmup calls per callable (default reproduces the historical scan).",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=REPETITIONS,
        help="CUDA Event samples per callable; use more for a fresh measurement.",
    )
    parser.add_argument(
        "--skip-profile",
        action="store_true",
        help="Run the Ksel timing sweep without separate profiler sessions.",
    )
    args = parser.parse_args()
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.repetitions <= 0:
        parser.error("--repetitions must be positive")
    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is unavailable. Install a CUDA-enabled PyTorch build and run on an NVIDIA GPU."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trace_dir = str(args.output_dir / "profiler_traces")
    q, k, v, block_ids = make_smoke_fixture()
    validate_smoke_fixture(q, k, v, block_ids)
    print("Benchmark input contract passed")

    # 这是顶层实验入口：一次调用生成所有 Ksel 实验点的完整证据。
    # 当前只用两次预热和两个样本验证扫描接线，不能作为稳定性能结论。
    sweep_results = run_k_forward_sweep(
        q=q,
        k=k,
        v=v,
        ksel_values=KSEL_SWEEP,
        warmup=args.warmup,
        repetitions=args.repetitions,
    )

    metadata = {
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(DEVICE),
        "device": str(DEVICE),
        "dtype": str(DTYPE),
        "shape": {"B": B, "N": N, "Hq": HQ, "Hkv": HKV, "D": D, "block_size": BLOCK_SIZE},
        "warmup": args.warmup,
        "repetitions": args.repetitions,
        "timing_method": "CUDA Events around the complete callable",
        "warning": "The default two-sample run is diagnostic; inspect each run\u2019s recorded settings and raw samples.",
    }
    result_path = args.output_dir / "benchmark_results.json"
    result_path.write_text(
        json.dumps(
            {"metadata": metadata, "sweep": sweep_results},
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    print(f"Environment and timing results: {result_path}")

    # 展示层只读取结果记录，不在打印过程中重新计时或计算支持集。
    for ksel, row in sweep_results.items():
        print(f"Ksel={ksel}, 有效支持集={row['support']}")
        for backend_name, samples_ms in row["samples_ms"].items():
            print(f"    {backend_name}: {samples_ms}")
            for summary_name, summary_value in row["summary_ms"][backend_name].items():
                print(f"        {summary_name}: {summary_value}")

    # profiler 与 Ksel 扫描是两个独立实验。这里用 PROFILE_KSEL 重新
    # 构造一份代表 metadata，而不使用 make_smoke_fixture() 返回的固定
    # block_ids，也不假设 KSEL_SWEEP 一定包含 2。
    profile_block_ids = make_prefix_block_ids(ksel=PROFILE_KSEL)

    # 正确性门禁位于 profiler 之外：只有三种 callable 能运行，并且
    # masked/gathered 依然数值等价，后面的成本归因才有解释价值。
    run_smoke_forward(
        q=q,
        k=k,
        v=v,
        block_ids=profile_block_ids,
    )

    # 三种 backend 使用相同 Q/K/V、profile_block_ids 和 profiler 配置，
    # 但各自进入独立 profiler session。这里产生的表格和 trace 只用于
    # 解释成本结构，不覆盖前面 CUDA Event 基准得到的延迟。
    if args.skip_profile:
        print("Skipped Profiler via --skip-profile; CUDA Event results were saved.")
        return

    profile_results = run_forward_profiling(
        q=q,
        k=k,
        v=v,
        block_ids=profile_block_ids,
        warmup=PROFILE_WARMUP,
        wait_steps=PROFILE_WAIT_STEPS,
        schedule_warmup_steps=PROFILE_SCHEDULE_WARMUP_STEPS,
        active_steps=PROFILE_ACTIVE_STEPS,
        repeat=PROFILE_REPEAT,
        row_limit=PROFILE_ROW_LIMIT,
        stack_depth=PROFILE_STACK_DEPTH,
        trace_dir=trace_dir,
    )

    table_dir = args.output_dir / "profiler_tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    profile_manifest = {}
    for backend_name, tables in profile_results.items():
        profile_manifest[backend_name] = {"trace_path": tables["trace_path"]}
        for table_name in (
            "cuda_table", "cpu_table", "memory_table", "flops_table", "stack_table"
        ):
            table_path = table_dir / f"{backend_name}_{table_name}.txt"
            table_path.write_text(tables[table_name] + "\n", encoding="utf-8")
            profile_manifest[backend_name][table_name] = table_path.as_posix()
    profile_path = args.output_dir / "profiler_results.json"
    profile_path.write_text(
        json.dumps(profile_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Profiler tables and trace index: {profile_path}")

    # 这里只展示已生成的字符串和文件路径，不会重新执行 backend。
    # 每个 backend 依次展示 CUDA、CPU、内存、FLOPs、调用栈五种聚合
    # 视图，最后打印保留原始时间线的 Chrome trace JSON 路径。
    for backend_name, tables in profile_results.items():
        print(f"\n===== {backend_name} =====")
        print(tables["cuda_table"])
        print(tables["cpu_table"])
        print(tables["memory_table"])
        print(tables["flops_table"])
        print(tables["stack_table"])
        print(f"Chrome trace: {tables['trace_path']}")


if __name__ == "__main__":
    main()


# endregion
