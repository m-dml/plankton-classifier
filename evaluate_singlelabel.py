import subprocess
from src.utils.eval_single_helper_funcs import get_best_checkpoints

if __name__ == "__main__":

    experiments_paths = {
        "supervised_singlelabel": [
            "/gpfs/work/machnitz/plankton_logs/supervised/singlelabel/multirun/2022-04-19/08-14-50"
        ],
        "finetune_singlelabel": [
          "/gpfs/work/machnitz/plankton_logs/finetune/singlelabel/multirun/2022-04-19/08-15-24"
        ],
        "linear_singlelabel": [
          "/gpfs/work/machnitz/plankton_logs/linear_eval/singlelabel/multirun/2022-04-29/08-31-47",
          "/gpfs/work/machnitz/plankton_logs/linear_eval/singlelabel/multirun/2022-05-02/10-33-55"
        ],
        "finetune_sgd_singlelabel": [
            "/gpfs/work/machnitz/plankton_logs/finetune_sgd/singlelabel/multirun/2022-04-29/08-19-35"
        ],
    }

    experiments = []
    for key, paths in experiments_paths.items():
        for path in paths:
            experiments += get_best_checkpoints(key, path)

    for experiment_str in experiments:
        experiment_str = '"' + experiment_str + '"'
        subprocess.Popen(["python",
                          "main.py",
                          "-m",
                          "+experiment=plankton/publication/evaluate_singlelabel",
                          "hydra/launcher=strand_single",
                          f"load_state_dict={experiment_str}",
                          ])
