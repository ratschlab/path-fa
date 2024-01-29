seeds = [117, 711, 420, 523, 187, 982, 766, 233, 523, 832]
seeds += [e+1 for e in seeds]
base_cmd = 'python run_synthetic.py'
for seed in seeds:
    print(base_cmd, '-s', seed)
