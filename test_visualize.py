"""
FermiNet 可视化模块 (visualize.py) 全面单元测试
=================================================
覆盖：学术样式设定、统计辅助函数、体系参考能量推断、
训练过程图表（能量收敛、方差/SEM、MCMC 诊断、Dashboard）、
训练结果图表（3D/2D 电子密度、局域能量直方图、波函数截面、物理 Dashboard）、
一键全自动可视化导出器。
"""

import os
import sys
import shutil
import tempfile
import unittest
from typing import Dict, List
import numpy as np
import torch

# 确保包路径可见
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
for p in [current_dir, parent_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

# 强制无 GUI 后端
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from visualize import (
    COLOR_PALETTE,
    REFERENCE_ENERGIES,
    apply_academic_style,
    compute_distribution_stats,
    get_system_reference,
    get_system_reference_energy,
    get_system_label,
    plot_energy_convergence,
    plot_variance_and_sem,
    plot_mcmc_diagnostics,
    plot_training_dashboard,
    plot_electron_density_3d,
    plot_electron_density_2d,
    plot_local_energy_distribution,
    plot_physical_results_dashboard,
    plot_all_visualizations,
)
from model import FermiNet
from hamiltonian import MolecularSystem


def _make_h2_system(device: torch.device = torch.device("cpu")) -> MolecularSystem:
    """创建用于测试的 H2 体系。"""
    nuclear_coords = torch.tensor(
        [[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]],
        dtype=torch.float32,
        device=device,
    )
    nuclear_charges = torch.tensor([1.0, 1.0], dtype=torch.float32, device=device)
    return MolecularSystem(
        nuclear_coords=nuclear_coords,
        nuclear_charges=nuclear_charges,
        n_up=1,
        n_down=1,
    )


def _make_training_history(n_epochs: int = 30) -> Dict[str, List[float]]:
    """构造模拟训练历史记录。"""
    np.random.seed(42)
    base_energy = -1.0
    history = {
        "energy_raw": [],
        "energy_clipped": [],
        "energy_std": [],
        "energy_var": [],
        "energy_iid_sem_proxy": [],
        "loss": [],
        "acceptance_rate": [],
        "step_size": [],
    }
    for i in range(n_epochs):
        progress = i / n_epochs
        e = base_energy - 0.17 * progress + np.random.randn() * 0.02 * (1 - progress)
        var = 0.5 * (1 - 0.9 * progress) + np.random.rand() * 0.02
        sem = np.sqrt(var / 1024)
        history["energy_raw"].append(float(e))
        history["energy_clipped"].append(float(e + np.random.randn() * 0.005))
        history["energy_std"].append(float(np.sqrt(var)))
        history["energy_var"].append(float(var))
        history["energy_iid_sem_proxy"].append(float(sem))
        history["loss"].append(float(abs(e) * 0.1))
        history["acceptance_rate"].append(float(0.4 + 0.1 * progress + np.random.rand() * 0.05))
        history["step_size"].append(float(0.5 - 0.15 * progress))
    return history


class TestAcademicStyleAndConstants(unittest.TestCase):
    """测试学术样式设定和常量定义。"""

    def test_color_palette_keys_present(self):
        """配色方案应包含所有必需键。"""
        required = [
            "raw_energy", "clipped_energy", "ref_energy", "sem_band",
            "variance", "sem", "acceptance", "step_size", "spin_up",
            "spin_down", "nucleus", "wavefunction", "cusp_marker",
        ]
        for key in required:
            self.assertIn(key, COLOR_PALETTE, f"配色方案缺少键: {key}")

    def test_reference_energies_values(self):
        """理论参考能量应为负值 (束缚态)。"""
        for name, val in REFERENCE_ENERGIES.items():
            self.assertLess(val, 0.0, f"参考能量 {name} 应为负值 (束缚态)")

    def test_h2_reference_energy(self):
        """H2 参考能量应约为 -1.174 Ha。"""
        self.assertAlmostEqual(REFERENCE_ENERGIES["h2"], -1.17447, places=3)

    def test_lih_reference_energy(self):
        """LiH 参考能量应约为 -8.070 Ha。"""
        self.assertAlmostEqual(REFERENCE_ENERGIES["lih"], -8.07050, places=3)

    def test_apply_academic_style_runs(self):
        """学术样式应用应无异常。"""
        apply_academic_style()
        self.assertEqual(plt.rcParams["savefig.dpi"], 300)


class TestStatisticsHelper(unittest.TestCase):
    """测试描述性统计辅助函数。"""

    def test_basic_stats_numpy(self):
        """NumPy 数组输入应正确计算均值/中位数/标准差。"""
        vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        stats = compute_distribution_stats(vals)
        self.assertEqual(stats["count"], 5)
        self.assertAlmostEqual(stats["mean"], 3.0, places=5)
        self.assertAlmostEqual(stats["median"], 3.0, places=5)

    def test_basic_stats_tensor(self):
        """PyTorch Tensor 输入应正确计算。"""
        vals = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        stats = compute_distribution_stats(vals)
        self.assertAlmostEqual(stats["mean"], 3.0, places=5)

    def test_basic_stats_list(self):
        """Python 列表输入应正确计算。"""
        vals = [10.0, 20.0, 30.0]
        stats = compute_distribution_stats(vals)
        self.assertAlmostEqual(stats["mean"], 20.0, places=5)

    def test_nan_filtering(self):
        """包含 NaN/Inf 的数据应被正确过滤。"""
        vals = np.array([1.0, float("nan"), 3.0, float("inf"), 5.0])
        stats = compute_distribution_stats(vals)
        self.assertEqual(stats["count"], 3)
        self.assertAlmostEqual(stats["mean"], 3.0, places=5)

    def test_all_nan_returns_zeros(self):
        """全 NaN 输入应返回零值字典。"""
        vals = np.array([float("nan"), float("nan")])
        stats = compute_distribution_stats(vals)
        self.assertEqual(stats["count"], 0)
        self.assertAlmostEqual(stats["mean"], 0.0)

    def test_skewness_symmetric(self):
        """对称分布的偏度应接近 0。"""
        vals = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        stats = compute_distribution_stats(vals)
        self.assertAlmostEqual(stats["skewness"], 0.0, places=3)

    def test_mad_computation(self):
        """MAD 应等于中位数绝对偏差。"""
        vals = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
        stats = compute_distribution_stats(vals)
        expected_mad = np.median(np.abs(vals - np.median(vals)))
        self.assertAlmostEqual(stats["mad"], expected_mad, places=5)


class TestSystemReferenceInference(unittest.TestCase):
    """测试分子体系自动推断函数。"""

    def test_string_lookup(self):
        """字符串名称应匹配到正确的参考能量。"""
        self.assertAlmostEqual(get_system_reference_energy("h2"), -1.17447, places=3)
        self.assertAlmostEqual(get_system_reference_energy("LiH"), -8.07050, places=3)

    def test_unknown_string_returns_none(self):
        """未知体系名称应返回 None。"""
        self.assertIsNone(get_system_reference_energy("unknown_molecule"))

    def test_none_input_returns_none(self):
        """None 输入应返回 None。"""
        self.assertIsNone(get_system_reference_energy(None))

    def test_molecular_system_h2(self):
        """H2 MolecularSystem 应自动推断为 H2 参考能。"""
        sys = _make_h2_system()
        ref = get_system_reference_energy(sys)
        self.assertAlmostEqual(ref, -1.17447, places=3)

    def test_reference_match_is_rigid_motion_and_permutation_invariant(self):
        """共同旋转、平移及原子重排不能改变体系身份。"""
        sys = _make_h2_system()
        angle = np.pi / 3.0
        rotation = torch.tensor(
            [[np.cos(angle), -np.sin(angle), 0.0],
             [np.sin(angle), np.cos(angle), 0.0],
             [0.0, 0.0, 1.0]],
            dtype=sys.nuclear_coords.dtype,
        )
        coords = (sys.nuclear_coords @ rotation.T) + torch.tensor([2.0, -1.0, 0.5])
        transformed = MolecularSystem(
            nuclear_coords=coords.flip(0),
            nuclear_charges=sys.nuclear_charges.flip(0),
            n_up=1,
            n_down=1,
        )
        ref = get_system_reference(transformed)
        self.assertIsNotNone(ref)
        self.assertEqual(ref.key, "h2")

    def test_reference_rejects_same_particle_counts_for_different_system(self):
        """两个核、两个电子不能自动被当作 H2。"""
        impostor = MolecularSystem(
            nuclear_coords=torch.tensor([[0.0, 0.0, -2.0], [0.0, 0.0, 2.0]]),
            nuclear_charges=torch.tensor([2.0, 1.0]),
            n_up=1,
            n_down=1,
        )
        self.assertIsNone(get_system_reference(impostor))
        self.assertIsNone(get_system_reference_energy(impostor))
        self.assertNotIn("H_2", get_system_label(impostor))

    def test_reference_rejects_wrong_geometry_and_spin(self):
        wrong_geometry = _make_h2_system()
        wrong_geometry.update_geometry(torch.tensor([[0.0, 0.0, -0.8], [0.0, 0.0, 0.8]]))
        wrong_spin = MolecularSystem(
            nuclear_coords=torch.tensor([[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]]),
            nuclear_charges=torch.tensor([1.0, 1.0]),
            n_up=2,
            n_down=0,
        )
        self.assertIsNone(get_system_reference_energy(wrong_geometry))
        self.assertIsNone(get_system_reference_energy(wrong_spin))

    def test_get_system_label_h2(self):
        """H2 体系标签应包含 H_2。"""
        sys = _make_h2_system()
        label = get_system_label(sys)
        self.assertIn("H", label)

    def test_get_system_label_string(self):
        """字符串输入应转大写。"""
        label = get_system_label("h2")
        self.assertEqual(label, "H2")

    def test_get_system_label_none(self):
        """None 输入应返回默认标签。"""
        label = get_system_label(None)
        self.assertEqual(label, "Molecule")


class TestTrainingProcessPlots(unittest.TestCase):
    """测试训练过程可视化绘图函数。"""

    def setUp(self):
        self.history = _make_training_history(20)
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        plt.close("all")
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_energy_convergence_returns_figure(self):
        """能量收敛图应返回 Figure 对象。"""
        fig = plot_energy_convergence(self.history, system_name="H2", ref_energy=-1.17447)
        self.assertIsInstance(fig, plt.Figure)

    def test_energy_convergence_save(self):
        """能量收敛图应正确保存为 PNG 文件。"""
        path = os.path.join(self.tmp_dir, "energy.png")
        fig = plot_energy_convergence(self.history, save_path=path)
        self.assertTrue(os.path.isfile(path))
        self.assertGreater(os.path.getsize(path), 1000)

    def test_energy_convergence_empty_history(self):
        """空历史记录应安全绘图 (不崩溃)。"""
        empty = {"energy_raw": [], "energy_clipped": [], "energy_iid_sem_proxy": []}
        fig = plot_energy_convergence(empty)
        self.assertIsInstance(fig, plt.Figure)

    def test_energy_convergence_with_custom_ax(self):
        """传入外部 Axes 应正确复用。"""
        fig, ax = plt.subplots()
        result_fig = plot_energy_convergence(self.history, ax=ax)
        self.assertIs(result_fig, fig)

    def test_variance_sem_returns_figure(self):
        """方差/SEM 图应返回 Figure 对象。"""
        fig = plot_variance_and_sem(self.history)
        self.assertIsInstance(fig, plt.Figure)

    def test_variance_sem_save(self):
        """方差/SEM 图应正确保存。"""
        path = os.path.join(self.tmp_dir, "var_sem.png")
        fig = plot_variance_and_sem(self.history, save_path=path)
        self.assertTrue(os.path.isfile(path))

    def test_variance_sem_log_scale(self):
        """对数坐标模式应无异常。"""
        fig = plot_variance_and_sem(self.history, log_scale=True)
        self.assertIsInstance(fig, plt.Figure)

    def test_variance_sem_linear_scale(self):
        """线性坐标模式应无异常。"""
        fig = plot_variance_and_sem(self.history, log_scale=False)
        self.assertIsInstance(fig, plt.Figure)

    def test_mcmc_diagnostics_returns_figure(self):
        """MCMC 诊断图应返回 Figure 对象。"""
        fig = plot_mcmc_diagnostics(self.history)
        self.assertIsInstance(fig, plt.Figure)

    def test_mcmc_diagnostics_custom_target(self):
        """自定义目标接受率应正确标注。"""
        fig = plot_mcmc_diagnostics(self.history, target_acc=0.6)
        self.assertIsInstance(fig, plt.Figure)

    def test_mcmc_diagnostics_save(self):
        """MCMC 诊断图应正确保存。"""
        path = os.path.join(self.tmp_dir, "mcmc.png")
        fig = plot_mcmc_diagnostics(self.history, save_path=path)
        self.assertTrue(os.path.isfile(path))

    def test_training_dashboard_returns_figure(self):
        """训练全景看板应返回 Figure 对象。"""
        fig = plot_training_dashboard(self.history, system_name="H2", ref_energy=-1.17447)
        self.assertIsInstance(fig, plt.Figure)

    def test_training_dashboard_save(self):
        """训练看板应正确保存。"""
        path = os.path.join(self.tmp_dir, "dashboard.png")
        fig = plot_training_dashboard(self.history, save_path=path)
        self.assertTrue(os.path.isfile(path))
        self.assertGreater(os.path.getsize(path), 5000)


class TestPhysicalStatePlots(unittest.TestCase):
    """测试训练结果物理状态可视化。"""

    def setUp(self):
        self.system = _make_h2_system()
        self.walkers = torch.randn(64, 2, 3)  # (B=64, N=2, 3)
        self.local_energies = torch.randn(64) * 0.1 - 1.0  # 围绕 -1.0 Ha
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        plt.close("all")
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_electron_density_3d_returns_figure(self):
        """3D 电子密度散点图应返回 Figure 对象。"""
        fig = plot_electron_density_3d(
            walkers=self.walkers,
            nuclear_coords=self.system.nuclear_coords,
            nuclear_charges=self.system.nuclear_charges,
            n_up=1, n_down=1,
        )
        self.assertIsInstance(fig, plt.Figure)

    def test_electron_density_3d_save(self):
        """3D 电子密度图应正确保存。"""
        path = os.path.join(self.tmp_dir, "density_3d.png")
        fig = plot_electron_density_3d(
            walkers=self.walkers,
            nuclear_coords=self.system.nuclear_coords,
            n_up=1, n_down=1,
            save_path=path,
        )
        self.assertTrue(os.path.isfile(path))

    def test_electron_density_3d_auto_spin_split(self):
        """未指定自旋数量时应自动推断。"""
        fig = plot_electron_density_3d(
            walkers=self.walkers,
            nuclear_coords=self.system.nuclear_coords,
        )
        self.assertIsInstance(fig, plt.Figure)

    def test_electron_density_3d_downsampling(self):
        """大量采样点应正确下采样而不崩溃。"""
        large_walkers = torch.randn(5000, 2, 3)
        fig = plot_electron_density_3d(
            walkers=large_walkers,
            nuclear_coords=self.system.nuclear_coords,
            max_points=500,
        )
        self.assertIsInstance(fig, plt.Figure)

    def test_electron_density_2d_returns_figure(self):
        """2D 电子密度图应返回 Figure 对象。"""
        fig = plot_electron_density_2d(
            walkers=self.walkers,
            nuclear_coords=self.system.nuclear_coords,
            n_up=1, n_down=1,
        )
        self.assertIsInstance(fig, plt.Figure)

    def test_electron_density_2d_save(self):
        """2D 电子密度图应正确保存。"""
        path = os.path.join(self.tmp_dir, "density_2d.png")
        fig = plot_electron_density_2d(
            walkers=self.walkers,
            nuclear_coords=self.system.nuclear_coords,
            save_path=path,
        )
        self.assertTrue(os.path.isfile(path))

    def test_local_energy_distribution_returns_figure(self):
        """局域能量直方图应返回 Figure 对象。"""
        fig = plot_local_energy_distribution(self.local_energies)
        self.assertIsInstance(fig, plt.Figure)

    def test_local_energy_distribution_with_ref(self):
        """有理论参考线的局域能量直方图应正常。"""
        fig = plot_local_energy_distribution(
            self.local_energies,
            ref_energy=-1.17447,
        )
        self.assertIsInstance(fig, plt.Figure)

    def test_local_energy_distribution_save(self):
        """局域能量直方图应正确保存。"""
        path = os.path.join(self.tmp_dir, "el_dist.png")
        fig = plot_local_energy_distribution(
            self.local_energies,
            save_path=path,
        )
        self.assertTrue(os.path.isfile(path))

    def test_local_energy_distribution_numpy_input(self):
        """NumPy 数组输入应正常。"""
        fig = plot_local_energy_distribution(
            np.random.randn(100) - 1.0,
        )
        self.assertIsInstance(fig, plt.Figure)

    def test_local_energy_distribution_list_input(self):
        """Python 列表输入应正常。"""
        fig = plot_local_energy_distribution(
            [-1.1, -1.2, -1.0, -1.15, -1.05],
        )
        self.assertIsInstance(fig, plt.Figure)

    def test_local_energy_distribution_empty(self):
        """空数据应安全处理。"""
        fig = plot_local_energy_distribution(
            np.array([float("nan"), float("nan")]),
        )
        self.assertIsInstance(fig, plt.Figure)

    def test_physical_dashboard_returns_figure(self):
        """物理状态看板应返回 Figure 对象。"""
        fig = plot_physical_results_dashboard(
            walkers=self.walkers,
            system=self.system,
            local_energies=self.local_energies,
        )
        self.assertIsInstance(fig, plt.Figure)

    def test_physical_dashboard_save(self):
        """物理状态看板应正确保存。"""
        path = os.path.join(self.tmp_dir, "phys_dash.png")
        fig = plot_physical_results_dashboard(
            walkers=self.walkers,
            system=self.system,
            local_energies=self.local_energies,
            save_path=path,
        )
        self.assertTrue(os.path.isfile(path))


class TestAllVisualizationsExporter(unittest.TestCase):
    """测试一键全自动可视化导出器。"""

    def setUp(self):
        self.system = _make_h2_system()
        self.history = _make_training_history(15)
        self.walkers = torch.randn(32, 2, 3)
        self.local_energies = torch.randn(32) * 0.1 - 1.0
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        plt.close("all")
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_exports_training_plots(self):
        """应导出所有训练过程图表。"""
        saved = plot_all_visualizations(
            history=self.history,
            system=self.system,
            walkers=self.walkers,
            local_energies=self.local_energies,
            plot_dir=self.tmp_dir,
        )
        self.assertIn("training_dashboard", saved)
        self.assertIn("energy_convergence", saved)
        self.assertIn("variance_sem", saved)
        self.assertIn("mcmc_diagnostics", saved)
        for key, path in saved.items():
            self.assertTrue(os.path.isfile(path), f"图表文件缺失: {key} -> {path}")

    def test_exports_electron_density_plots(self):
        """应导出电子密度分布图。"""
        saved = plot_all_visualizations(
            history=self.history,
            system=self.system,
            walkers=self.walkers,
            local_energies=self.local_energies,
            plot_dir=self.tmp_dir,
        )
        self.assertIn("electron_density_3d", saved)
        self.assertIn("electron_density_2d", saved)

    def test_exports_local_energy_dist(self):
        """应导出局域能量直方图。"""
        saved = plot_all_visualizations(
            history=self.history,
            system=self.system,
            local_energies=self.local_energies,
            plot_dir=self.tmp_dir,
        )
        self.assertIn("local_energy_dist", saved)

    def test_exports_physical_dashboard(self):
        """应导出物理状态全景看板。"""
        saved = plot_all_visualizations(
            history=self.history,
            system=self.system,
            walkers=self.walkers,
            local_energies=self.local_energies,
            plot_dir=self.tmp_dir,
        )
        self.assertIn("physical_dashboard", saved)

    def test_prefix_in_filenames(self):
        """文件名应包含前缀。"""
        saved = plot_all_visualizations(
            history=self.history,
            system=self.system,
            walkers=self.walkers,
            local_energies=self.local_energies,
            plot_dir=self.tmp_dir,
            prefix="h2",
        )
        for key, path in saved.items():
            basename = os.path.basename(path)
            self.assertTrue(basename.startswith("h2_"), f"文件名 {basename} 缺少前缀 'h2_'")

    def test_minimal_history_only(self):
        """仅提供 history (无 walkers/energies) 应只导出训练图表。"""
        saved = plot_all_visualizations(
            history=self.history,
            system=self.system,
            plot_dir=self.tmp_dir,
        )
        self.assertIn("training_dashboard", saved)
        self.assertNotIn("electron_density_3d", saved)
        self.assertNotIn("local_energy_dist", saved)

    def test_empty_history_no_crash(self):
        """空历史记录不应导致崩溃。"""
        empty = {"energy_raw": [], "energy_var": [], "energy_iid_sem_proxy": [],
                 "acceptance_rate": [], "step_size": []}
        saved = plot_all_visualizations(
            history=empty,
            system=self.system,
            plot_dir=self.tmp_dir,
        )
        # 空历史不应生成训练图表
        self.assertNotIn("training_dashboard", saved)

    def test_all_files_non_zero_size(self):
        """所有导出文件应有非零大小 (图片内容实质性存在)。"""
        saved = plot_all_visualizations(
            history=self.history,
            system=self.system,
            walkers=self.walkers,
            local_energies=self.local_energies,
            plot_dir=self.tmp_dir,
        )
        for key, path in saved.items():
            self.assertGreater(
                os.path.getsize(path), 1000,
                f"图表文件 {key} 大小异常 (< 1KB): {os.path.getsize(path)} bytes"
            )


if __name__ == "__main__":
    unittest.main()
