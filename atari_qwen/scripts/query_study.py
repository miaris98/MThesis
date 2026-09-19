import optuna

storage = "sqlite:////workspace/MThesis/results/atari_gtrxl_optuna.db"
study = optuna.load_study(study_name="gtrxl_ez2_hparam_study", storage=storage)
print(f"=== Study: {study.study_name} ===")
print(f"Total Trials: {len(study.trials)}")
for t in study.trials:
    last_step = max(t.intermediate_values.keys()) if t.intermediate_values else 0
    print(f"Trial #{t.number:02d} | State: {t.state.name:10s} | Value: {t.value} | Last Step: {last_step}")
    if t.intermediate_values:
        print(f"   Intermediates: {t.intermediate_values}")
