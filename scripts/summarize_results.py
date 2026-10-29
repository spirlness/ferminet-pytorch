import json

EXACT_ENERGY = -1.174475714220
HF_ENERGY = -1.13362957


def main():
    for env_name, path in [
        ("CUDA (RTX 3060+)", "m4_convergence_cuda.json"),
        ("CPU (uv 2.14.0+cpu)", "m4_convergence_cpu.json"),
    ]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print("=" * 100)
        print(
            f"### 环境: {env_name} (H2, R=1.4 bohr, Exact E = {EXACT_ENERGY:.6f} Ha, HF = {HF_ENERGY:.6f} Ha)"
        )
        print("=" * 100)
        for exp_k, exp_v in data.items():
            name = exp_v["experiment_name"]
            best_e = exp_v["best_energy"]
            best_ep = exp_v["best_epoch"]
            final_e = exp_v["final_energy"]
            final_v = exp_v["final_var"]
            final_sem = exp_v["final_sem"]
            t_tot = exp_v["total_train_time"]
            eval_res = exp_v["eval_result"]
            eval_mean = eval_res["energy_mean"]
            eval_sem = eval_res["energy_sem"]
            eval_var = eval_res["energy_var"]
            eval_ess = eval_res["effective_sample_size"]
            abs_err_mha = eval_res.get(
                "abs_error_mha", abs(eval_mean - EXACT_ENERGY) * 1000
            )

            print(f"\n[{name}]")
            print(
                f"  - 模型类: {exp_v['model_class']} | 训练总耗时: {t_tot:.2f}s ({t_tot / 15:.3f}s/epoch)"
            )
            print(
                f"  - 训练收敛: 历史最优 E = {best_e:.5f} Ha (Epoch {best_ep}) | 第 15 轮 E = {final_e:.5f} Ha, Var = {final_v:.5f}, SEM = {final_sem:.5f}"
            )
            print(
                f"  - 冷冻评估 (512 walkers * 30 steps): E = {eval_mean:.6f} +/- {eval_sem:.6f} Ha | Var = {eval_var:.5f} | ESS = {int(eval_ess)} | 距精确解 = {abs_err_mha:.2f} mHa"
            )

            # 打印逐轮收敛轨迹
            print("  - 训练详细轨迹 (前5轮与后5轮):")
            logs = exp_v["epoch_logs"]
            print(
                f"    {'Epoch':^5} | {'<E_raw> (Ha)':^12} | {'<E_clip> (Ha)':^12} | {'Var(E)':^8} | {'SEM':^8} | {'MAD':^7} | {'Clip#':^6} | {'Acc (%)':^7} | {'Step':^6} | {'GradNorm':^8}"
            )
            for l in logs:
                ep = l["epoch"]
                if ep <= 5 or ep >= 11:
                    print(
                        f"    {ep:^5d} | {l['energy_raw']:^12.5f} | {l['energy_clipped']:^12.5f} | {l['energy_var']:^8.4f} | {l['energy_sem']:^8.4f} | {l['energy_mad']:^7.4f} | {l['clipped_samples']:^6d} | {l['mcmc_acc'] * 100:^7.1f} | {l['step_size']:^6.4f} | {l['grad_norm']:^8.4f}"
                    )
                elif ep == 6:
                    print("    ... (Epoch 6~10 省略) ...")


if __name__ == "__main__":
    main()
