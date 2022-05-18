import subprocess
from src.utils.eval_single_helper_funcs import get_best_checkpoints

if __name__ == "__main__":

    experiments_paths = {
        # "fintune_frozen": [
        #     "/gpfs/work/machnitz/plankton_logs/finetune_frozen/singlelabel/multirun/2022-05-06/14-46-05"
        # ],
        # "supervised_singlelabel": [
        #     "/gpfs/work/machnitz/plankton_logs/supervised/singlelabel/multirun/2022-05-16/09-45-24"
        # ],
        # "finetune_singlelabel": [
        #     "/gpfs/work/machnitz/plankton_logs/finetune/singlelabel/multirun/2022-05-05/07-02-47"
        # ],
        # "linear_singlelabel": [
        #     "/gpfs/work/machnitz/plankton_logs/linear_eval/singlelabel/multirun/2022-05-06/13-58-58",
        # ],
        "finetune_sgd_singlelabel": [
            "/gpfs/work/machnitz/plankton_logs/finetune_sgd/singlelabel/multirun/2022-05-17/07-38-08",
            "/gpfs/work/machnitz/plankton_logs/finetune_sgd/singlelabel/multirun/2022-05-18/07-01-20"
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
