import subprocess

if __name__ == "__main__":
    
    experiments = [
        "/gpfs/work/machnitz/plankton_logs/supervised/singlelabel/multirun/2022-04-19/08-14-50/0/logs/checkpoints/epoch=99.ckpt",
        "/gpfs/work/machnitz/plankton_logs/supervised/singlelabel/multirun/2022-04-19/08-14-50/1/logs/checkpoints/epoch=47.ckpt",
        '/gpfs/work/machnitz/plankton_logs/supervised/singlelabel/multirun/2022-04-19/08-14-50/2/logs/checkpoints/epoch=37.ckpt',
        '/gpfs/work/machnitz/plankton_logs/supervised/singlelabel/multirun/2022-04-19/08-14-50/3/logs/checkpoints/epoch=26.ckpt',
        '/gpfs/work/machnitz/plankton_logs/supervised/singlelabel/multirun/2022-04-19/08-14-50/4/logs/checkpoints/epoch=19.ckpt',
        '/gpfs/work/machnitz/plankton_logs/supervised/singlelabel/multirun/2022-04-19/08-14-50/5/logs/checkpoints/epoch=15.ckpt',
        '/gpfs/work/machnitz/plankton_logs/supervised/singlelabel/multirun/2022-04-19/08-14-50/6/logs/checkpoints/epoch=14.ckpt',
        '/gpfs/work/machnitz/plankton_logs/supervised/singlelabel/multirun/2022-04-19/08-14-50/7/logs/checkpoints/epoch=16.ckpt',
        '/gpfs/work/machnitz/plankton_logs/supervised/singlelabel/multirun/2022-04-19/08-14-50/8/logs/checkpoints/epoch=14.ckpt',
        '/gpfs/work/machnitz/plankton_logs/supervised/singlelabel/multirun/2022-04-19/08-14-50/9/logs/checkpoints/epoch=13.ckpt',
        '/gpfs/work/machnitz/plankton_logs/supervised/singlelabel/multirun/2022-04-19/08-14-50/10/logs/checkpoints/epoch=04.ckpt',
        '/gpfs/work/machnitz/plankton_logs/supervised/singlelabel/multirun/2022-04-19/08-14-50/11/logs/checkpoints/epoch=04.ckpt',
        '/gpfs/work/machnitz/plankton_logs/supervised/singlelabel/multirun/2022-04-19/08-14-50/12/logs/checkpoints/epoch=02.ckpt',
        '/gpfs/work/machnitz/plankton_logs/supervised/singlelabel/multirun/2022-04-19/08-14-50/13/logs/checkpoints/epoch=02.ckpt',
        '/gpfs/work/machnitz/plankton_logs/supervised/singlelabel/multirun/2022-04-19/08-14-50/14/logs/checkpoints/epoch=02.ckpt',
        '/gpfs/work/machnitz/plankton_logs/supervised/singlelabel/multirun/2022-04-19/08-14-50/15/logs/checkpoints/epoch=01-v1.ckpt',
        '/gpfs/work/machnitz/plankton_logs/supervised/singlelabel/multirun/2022-04-19/08-14-50/16/logs/checkpoints/epoch=01-v1.ckpt',
        '/gpfs/work/machnitz/plankton_logs/supervised/singlelabel/multirun/2022-04-19/08-14-50/17/logs/checkpoints/epoch=01-v1.ckpt',
        '/gpfs/work/machnitz/plankton_logs/supervised/singlelabel/multirun/2022-04-19/08-14-50/18/logs/checkpoints/epoch=01.ckpt',
        '/gpfs/work/machnitz/plankton_logs/linear_eval/singlelabel/multirun/2022-04-19/19-15-20/0/logs/checkpoints/epoch=21.ckpt',
        '/gpfs/work/machnitz/plankton_logs/linear_eval/singlelabel/multirun/2022-04-19/19-15-20/1/logs/checkpoints/epoch=25.ckpt',
        '/gpfs/work/machnitz/plankton_logs/linear_eval/singlelabel/multirun/2022-04-19/19-15-20/2/logs/checkpoints/epoch=10.ckpt',
        '/gpfs/work/machnitz/plankton_logs/linear_eval/singlelabel/multirun/2022-04-19/19-15-20/3/logs/checkpoints/epoch=27.ckpt',
        '/gpfs/work/machnitz/plankton_logs/linear_eval/singlelabel/multirun/2022-04-19/19-15-20/4/logs/checkpoints/epoch=22.ckpt',
        '/gpfs/work/machnitz/plankton_logs/linear_eval/singlelabel/multirun/2022-04-19/19-15-20/5/logs/checkpoints/epoch=12.ckpt',
        '/gpfs/work/machnitz/plankton_logs/linear_eval/singlelabel/multirun/2022-04-19/19-15-20/6/logs/checkpoints/epoch=21.ckpt',
        '/gpfs/work/machnitz/plankton_logs/linear_eval/singlelabel/multirun/2022-04-19/19-15-20/7/logs/checkpoints/epoch=05.ckpt',
        '/gpfs/work/machnitz/plankton_logs/linear_eval/singlelabel/multirun/2022-04-19/19-15-20/8/logs/checkpoints/epoch=13.ckpt',
        '/gpfs/work/machnitz/plankton_logs/linear_eval/singlelabel/multirun/2022-04-19/19-15-20/9/logs/checkpoints/epoch=03.ckpt',
        '/gpfs/work/machnitz/plankton_logs/linear_eval/singlelabel/multirun/2022-04-19/19-15-20/10/logs/checkpoints/epoch=04.ckpt',
        '/gpfs/work/machnitz/plankton_logs/linear_eval/singlelabel/multirun/2022-04-19/19-15-20/11/logs/checkpoints/epoch=04.ckpt',
        '/gpfs/work/machnitz/plankton_logs/linear_eval/singlelabel/multirun/2022-04-19/19-15-20/12/logs/checkpoints/epoch=02.ckpt',
        '/gpfs/work/machnitz/plankton_logs/linear_eval/singlelabel/multirun/2022-04-19/19-15-20/13/logs/checkpoints/epoch=01.ckpt',
        '/gpfs/work/machnitz/plankton_logs/linear_eval/singlelabel/multirun/2022-04-19/19-15-20/14/logs/checkpoints/epoch=02.ckpt',
        '/gpfs/work/machnitz/plankton_logs/linear_eval/singlelabel/multirun/2022-04-19/19-15-20/15/logs/checkpoints/epoch=01.ckpt',
        '/gpfs/work/machnitz/plankton_logs/linear_eval/singlelabel/multirun/2022-04-19/19-15-20/16/logs/checkpoints/epoch=00-v1.ckpt',
        '/gpfs/work/machnitz/plankton_logs/linear_eval/singlelabel/multirun/2022-04-19/19-15-20/17/logs/checkpoints/epoch=00.ckpt',
        '/gpfs/work/machnitz/plankton_logs/linear_eval/singlelabel/multirun/2022-04-19/19-15-20/18/logs/checkpoints/epoch=01.ckpt',
        '/gpfs/work/machnitz/plankton_logs/finetune/singlelabel/multirun/2022-04-19/08-15-24/0/logs/checkpoints/epoch=89.ckpt',
        '/gpfs/work/machnitz/plankton_logs/finetune/singlelabel/multirun/2022-04-19/08-15-24/1/logs/checkpoints/epoch=40.ckpt',
        '/gpfs/work/machnitz/plankton_logs/finetune/singlelabel/multirun/2022-04-19/08-15-24/2/logs/checkpoints/epoch=37.ckpt',
        '/gpfs/work/machnitz/plankton_logs/finetune/singlelabel/multirun/2022-04-19/08-15-24/3/logs/checkpoints/epoch=22.ckpt',
        '/gpfs/work/machnitz/plankton_logs/finetune/singlelabel/multirun/2022-04-19/08-15-24/4/logs/checkpoints/epoch=19.ckpt',
        '/gpfs/work/machnitz/plankton_logs/finetune/singlelabel/multirun/2022-04-19/08-15-24/5/logs/checkpoints/epoch=15.ckpt',
        '/gpfs/work/machnitz/plankton_logs/finetune/singlelabel/multirun/2022-04-19/08-15-24/6/logs/checkpoints/epoch=15.ckpt',
        '/gpfs/work/machnitz/plankton_logs/finetune/singlelabel/multirun/2022-04-19/08-15-24/7/logs/checkpoints/epoch=16.ckpt',
        '/gpfs/work/machnitz/plankton_logs/finetune/singlelabel/multirun/2022-04-19/08-15-24/8/logs/checkpoints/epoch=12.ckpt',
        '/gpfs/work/machnitz/plankton_logs/finetune/singlelabel/multirun/2022-04-19/08-15-24/9/logs/checkpoints/epoch=13.ckpt',
        '/gpfs/work/machnitz/plankton_logs/finetune/singlelabel/multirun/2022-04-19/08-15-24/10/logs/checkpoints/epoch=04.ckpt',
        '/gpfs/work/machnitz/plankton_logs/finetune/singlelabel/multirun/2022-04-19/08-15-24/11/logs/checkpoints/epoch=04.ckpt',
        '/gpfs/work/machnitz/plankton_logs/finetune/singlelabel/multirun/2022-04-19/08-15-24/12/logs/checkpoints/epoch=02.ckpt',
        '/gpfs/work/machnitz/plankton_logs/finetune/singlelabel/multirun/2022-04-19/08-15-24/13/logs/checkpoints/epoch=02.ckpt',
        '/gpfs/work/machnitz/plankton_logs/finetune/singlelabel/multirun/2022-04-19/08-15-24/14/logs/checkpoints/epoch=02.ckpt',
        '/gpfs/work/machnitz/plankton_logs/finetune/singlelabel/multirun/2022-04-19/08-15-24/15/logs/checkpoints/epoch=01-v1.ckpt',
        '/gpfs/work/machnitz/plankton_logs/finetune/singlelabel/multirun/2022-04-19/08-15-24/16/logs/checkpoints/epoch=01-v1.ckpt',
        '/gpfs/work/machnitz/plankton_logs/finetune/singlelabel/multirun/2022-04-19/08-15-24/17/logs/checkpoints/epoch=01-v1.ckpt',
        '/gpfs/work/machnitz/plankton_logs/finetune/singlelabel/multirun/2022-04-19/08-15-24/18/logs/checkpoints/epoch=01.ckpt',
        '/gpfs/work/machnitz/plankton_logs/linear_eval/singlelabel/multirun/2022-04-19/08-15-10/0/logs/checkpoints/epoch=04.ckpt',
        '/gpfs/work/machnitz/plankton_logs/linear_eval/singlelabel/multirun/2022-04-19/08-15-10/1/logs/checkpoints/epoch=07.ckpt',
        '/gpfs/work/machnitz/plankton_logs/linear_eval/singlelabel/multirun/2022-04-19/08-15-10/2/logs/checkpoints/epoch=05.ckpt',
        '/gpfs/work/machnitz/plankton_logs/linear_eval/singlelabel/multirun/2022-04-19/08-15-10/3/logs/checkpoints/epoch=15.ckpt',
        '/gpfs/work/machnitz/plankton_logs/linear_eval/singlelabel/multirun/2022-04-19/08-15-10/4/logs/checkpoints/epoch=01.ckpt',
        '/gpfs/work/machnitz/plankton_logs/linear_eval/singlelabel/multirun/2022-04-19/08-15-10/5/logs/checkpoints/epoch=08.ckpt',
        '/gpfs/work/machnitz/plankton_logs/linear_eval/singlelabel/multirun/2022-04-19/08-15-10/6/logs/checkpoints/epoch=08.ckpt',
        '/gpfs/work/machnitz/plankton_logs/linear_eval/singlelabel/multirun/2022-04-19/08-15-10/7/logs/checkpoints/epoch=04.ckpt',
        '/gpfs/work/machnitz/plankton_logs/linear_eval/singlelabel/multirun/2022-04-19/08-15-10/8/logs/checkpoints/epoch=07.ckpt',
        '/gpfs/work/machnitz/plankton_logs/linear_eval/singlelabel/multirun/2022-04-19/08-15-10/9/logs/checkpoints/epoch=03.ckpt',
        '/gpfs/work/machnitz/plankton_logs/linear_eval/singlelabel/multirun/2022-04-19/08-15-10/10/logs/checkpoints/epoch=04.ckpt',
        '/gpfs/work/machnitz/plankton_logs/linear_eval/singlelabel/multirun/2022-04-19/08-15-10/11/logs/checkpoints/epoch=04.ckpt',
        '/gpfs/work/machnitz/plankton_logs/linear_eval/singlelabel/multirun/2022-04-19/08-15-10/12/logs/checkpoints/epoch=02.ckpt',
        '/gpfs/work/machnitz/plankton_logs/linear_eval/singlelabel/multirun/2022-04-19/08-15-10/13/logs/checkpoints/epoch=01.ckpt',
        '/gpfs/work/machnitz/plankton_logs/linear_eval/singlelabel/multirun/2022-04-19/08-15-10/14/logs/checkpoints/epoch=02.ckpt',
        '/gpfs/work/machnitz/plankton_logs/linear_eval/singlelabel/multirun/2022-04-19/08-15-10/15/logs/checkpoints/epoch=01.ckpt',
        '/gpfs/work/machnitz/plankton_logs/linear_eval/singlelabel/multirun/2022-04-19/08-15-10/16/logs/checkpoints/epoch=00-v1.ckpt',
        '/gpfs/work/machnitz/plankton_logs/linear_eval/singlelabel/multirun/2022-04-19/08-15-10/17/logs/checkpoints/epoch=00.ckpt',
        '/gpfs/work/machnitz/plankton_logs/linear_eval/singlelabel/multirun/2022-04-19/08-15-10/18/logs/checkpoints/epoch=01.ckpt',
    ]

    experiment_str = ",".join(experiments)

    subprocess.Popen(["python",
                      "main.py",
                      "+experiment=plankton/publication/evaluate_singlelabel",
                      "hydra/launcher=strand_single",
                      f"load_state_dict={experiment_str}"
                      "-m"])