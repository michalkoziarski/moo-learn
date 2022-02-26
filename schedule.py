import os

if __name__ == "__main__":
    for max_iter in [100, 200, 500, 1000, 2000]:
        for n_layers in [1, 2, 3, 4]:
            for n_test_rays in [5, 10, 15, 20, 25, 50, 100]:
                command = f"sbatch run.sh trial.py -max_iter {max_iter} -n_layers {n_layers} -n_test_rays {n_test_rays}"

                os.system(command)
