"""Streamlit dashboard for graph scheduling simulations and benchmark reports."""

from __future__ import annotations

import json
import random
from typing import Any

import pandas as pd
import streamlit as st

from heterograph_scheduler.models import GraphTask, TaskKind
from heterograph_scheduler.simulation import (
    cpu_only_estimate,
    simulate,
    synthetic_resources,
)


def _cuda_device_name() -> str | None:
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_name(0)


def _make_metadata_tasks(
    count: int,
    min_nodes: int,
    max_nodes: int,
    average_degree: int,
    feature_dim: int,
    seed: int,
) -> tuple[GraphTask, ...]:
    rng = random.Random(seed)
    tasks = []
    for index in range(count):
        nodes = rng.randint(min_nodes, max_nodes)
        edge_scale = rng.uniform(0.75, 1.25)
        edges = max(1, int(nodes * average_degree * edge_scale))
        tasks.append(
            GraphTask(
                task_id=f"task-{index + 1:03d}",
                kind=TaskKind.NEIGHBOR_AGGREGATION,
                num_nodes=nodes,
                num_edges=edges,
                feature_dim=feature_dim,
                priority=rng.choice((1.0, 1.0, 1.25, 1.5)),
                degree_skew=rng.uniform(1.0, 3.0),
            )
        )
    return tuple(tasks)


def _render_header(cuda_name: str | None) -> None:
    st.title("异构图计算调度可视化平台")
    st.caption("图任务建模 · 资源适配 · 调度仿真 · 硬件基准结果分析")
    left, right = st.columns(2)
    left.metric("调度模式", "任务感知资源匹配")
    right.metric("CUDA设备", cuda_name or "未检测到（可查看已保存基准报告）")
    st.info(
        "仿真页面使用示例资源参数生成估算结果，不代表真实硬件性能。"
        "请在“硬件基准”页导入 benchmark_cuda.py 输出的 JSON，查看CPU/GPU实测对比。"
    )


def _render_simulation() -> None:
    st.subheader("任务调度仿真")
    with st.form("simulation_form"):
        col_a, col_b, col_c = st.columns(3)
        task_count = col_a.slider("任务数", min_value=4, max_value=40, value=12)
        min_nodes = col_b.number_input(
            "最小节点数", min_value=100, max_value=1_000_000, value=10_000, step=1_000
        )
        max_nodes = col_c.number_input(
            "最大节点数", min_value=100, max_value=1_000_000, value=100_000, step=1_000
        )
        col_d, col_e, col_f = st.columns(3)
        average_degree = col_d.slider("平均度（边/节点）", 1, 40, 10)
        feature_dim = col_e.selectbox("特征维度", (32, 64, 128, 256), index=2)
        seed = col_f.number_input("随机种子", min_value=0, max_value=1_000_000, value=42)
        policy = st.selectbox(
            "调度策略",
            ("heuristic", "dqn"),
            format_func=lambda value: (
                "启发式（估算延迟最小）"
                if value == "heuristic"
                else "DQN（在线实验策略，结果可能波动）"
            ),
        )

        st.markdown("#### 参与仿真的资源")
        resource_defaults = {resource.name: resource for resource in synthetic_resources()}
        resource_cols = st.columns(3)
        enabled = {
            "cpu": resource_cols[0].checkbox("CPU", value=True),
            "gpu": resource_cols[1].checkbox("GPU", value=True),
            "fpga": resource_cols[2].checkbox("FPGA（仿真）", value=False),
        }
        run_simulation = st.form_submit_button(
            "运行调度仿真", type="primary", use_container_width=True
        )

    if min_nodes > max_nodes:
        st.error("最小节点数不能大于最大节点数。")
        return
    if not run_simulation:
        st.caption("设置工作负载参数后，点击“运行调度仿真”。")
        return
    selected_resources = [
        resource_defaults[name]
        for name, is_enabled in enabled.items()
        if is_enabled
    ]
    if not selected_resources:
        st.error("至少选择一个参与仿真的资源。")
        return

    try:
        tasks = _make_metadata_tasks(
            int(task_count),
            int(min_nodes),
            int(max_nodes),
            average_degree,
            feature_dim,
            int(seed),
        )
        result = simulate(tasks, selected_resources, policy=policy)
        cpu_baseline = cpu_only_estimate(tasks, selected_resources)
    except (RuntimeError, ValueError) as exc:
        st.error(f"仿真失败：{exc}")
        return

    assignments = pd.DataFrame(result["assignments"])
    assignment_counts = (
        assignments.groupby("device", as_index=False)
        .size()
        .rename(columns={"size": "任务数"})
    )
    total_estimate = float(result["estimated_total_latency_ms"])
    speedup = cpu_baseline / total_estimate if total_estimate > 0 else 0.0
    metrics = st.columns(4)
    metrics[0].metric("任务数", int(result["task_count"]))
    metrics[1].metric("估算总耗时", f"{total_estimate:.3f} ms")
    metrics[2].metric("CPU-only估算", f"{cpu_baseline:.3f} ms")
    metrics[3].metric("相对CPU估算比", f"{speedup:.2f}×")

    left, right = st.columns(2)
    with left:
        st.markdown("#### 资源任务分配")
        st.bar_chart(assignment_counts.set_index("device"), y="任务数")
    with right:
        st.markdown("#### 单任务估算延迟")
        latency_chart = assignments.set_index("task_id")[["estimated_latency_ms"]]
        st.bar_chart(latency_chart)
    st.markdown("#### 调度明细")
    st.dataframe(assignments, use_container_width=True, hide_index=True)
    st.download_button(
        "下载仿真结果 JSON",
        data=json.dumps(result, ensure_ascii=False, indent=2),
        file_name="graph_schedule_simulation.json",
        mime="application/json",
    )
    if policy == "dqn":
        st.warning(
            "DQN目前是随机初始化并在线更新的实验原型，不是训练完成的策略。"
            "该仿真结果不能作为调度效果结论。"
        )


def _load_benchmark_report(uploaded_file: Any) -> dict[str, Any] | None:
    try:
        report = json.loads(uploaded_file.getvalue())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        st.error(f"无法解析JSON文件：{exc}")
        return None
    if not isinstance(report, dict) or not isinstance(report.get("cases"), list):
        st.error("报告格式不正确：需要包含 cases 数组的JSON对象。")
        return None
    required = {
        "nodes",
        "edges",
        "feature_dim",
        "cpu_torch_median_ms",
        "cuda_resident_median_ms",
        "cuda_end_to_end_median_ms",
    }
    if not report["cases"]:
        st.error("报告中的 cases 为空。")
        return None
    for index, case in enumerate(report["cases"]):
        if not isinstance(case, dict) or not required.issubset(case):
            st.error(
                f"第 {index + 1} 个测试案例缺少必要字段："
                f"{', '.join(sorted(required))}。"
            )
            return None
    return report


def _render_benchmark() -> None:
    st.subheader("CPU / CUDA 实测报告")
    st.write(
        "上传 `scripts/benchmark_cuda.py` 生成的JSON。此页面展示实际测量耗时；"
        "驻留计算数据与端到端数据分别呈现。"
    )
    uploaded_file = st.file_uploader("选择基准报告 JSON", type=("json",))
    if uploaded_file is None:
        st.code(
            "python scripts/benchmark_cuda.py --nodes 10000 50000 100000 "
            "--avg-degree 10 --features 64 128 --warmup 10 --repeats 50",
            language="bash",
        )
        return
    report = _load_benchmark_report(uploaded_file)
    if report is None:
        return

    st.caption(
        f"GPU：{report.get('gpu_name', '未记录')} · "
        f"PyTorch：{report.get('torch_version', '未记录')} · "
        f"CUDA：{report.get('torch_cuda_version', '未记录')} · "
        f"CPU线程：{report.get('cpu_threads', '未记录')}"
    )
    frame = pd.DataFrame(report["cases"])
    frame["case"] = frame.apply(
        lambda row: (
            f"{int(row['nodes']):,}节点 / {int(row['edges']):,}边 / "
            f"{int(row['feature_dim'])}维"
        ),
        axis=1,
    )
    frame["resident_speedup"] = (
        frame["cpu_torch_median_ms"] / frame["cuda_resident_median_ms"]
    )
    frame["end_to_end_speedup"] = (
        frame["cpu_torch_median_ms"] / frame["cuda_end_to_end_median_ms"]
    )

    selected_metric = st.radio(
        "图表指标",
        ("耗时（ms）", "相对CPU加速比"),
        horizontal=True,
    )
    if selected_metric == "耗时（ms）":
        chart = frame.set_index("case")[
            [
                "cpu_torch_median_ms",
                "cuda_resident_median_ms",
                "cuda_end_to_end_median_ms",
            ]
        ].rename(
            columns={
                "cpu_torch_median_ms": "CPU PyTorch",
                "cuda_resident_median_ms": "CUDA常驻",
                "cuda_end_to_end_median_ms": "CUDA端到端",
            }
        )
        st.bar_chart(chart)
    else:
        chart = frame.set_index("case")[
            ["resident_speedup", "end_to_end_speedup"]
        ].rename(
            columns={
                "resident_speedup": "CUDA常驻加速比",
                "end_to_end_speedup": "CUDA端到端加速比",
            }
        )
        st.bar_chart(chart)

    display_columns = [
        "nodes",
        "edges",
        "feature_dim",
        "cpu_torch_median_ms",
        "cuda_resident_median_ms",
        "cuda_end_to_end_median_ms",
        "resident_speedup",
        "end_to_end_speedup",
    ]
    st.dataframe(
        frame[display_columns].rename(
            columns={
                "nodes": "节点数",
                "edges": "边数",
                "feature_dim": "特征维度",
                "cpu_torch_median_ms": "CPU耗时(ms)",
                "cuda_resident_median_ms": "CUDA常驻(ms)",
                "cuda_end_to_end_median_ms": "CUDA端到端(ms)",
                "resident_speedup": "常驻加速比",
                "end_to_end_speedup": "端到端加速比",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(report.get("timing_note", "报告未提供计时说明。"))
    st.warning(
        "该报告衡量的是随机图上的 index_add_ 邻居聚合微基准，"
        "不是完整PyG模型训练性能，也不等同于真实业务图的最终加速比。"
    )


def main() -> None:
    st.set_page_config(
        page_title="异构图计算调度平台",
        page_icon="🔀",
        layout="wide",
    )
    cuda_name = _cuda_device_name()
    _render_header(cuda_name)
    simulation_tab, benchmark_tab = st.tabs(("调度仿真", "硬件基准"))
    with simulation_tab:
        _render_simulation()
    with benchmark_tab:
        _render_benchmark()


if __name__ == "__main__":
    main()
