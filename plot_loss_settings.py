import matplotlib.pyplot as plt
import json
import seaborn as sns
import os

sns.set_theme()


exps_dir = 'experiments/cifar10/0_kernel_size/alexnet'
exp_settings = os.listdir(exps_dir)
model = os.path.basename(exps_dir)

plt.figure()

for exp in exp_settings:
    sub_dir = os.path.join(exps_dir, exp)

    test_losses = []

    log_json_files = os.path.join(sub_dir, 'log.json')

    with open(log_json_files, 'r') as f:
        log = json.load(f)
        test_log = log['test']

    for key, val in test_log.items():
        test_losses.append(val['loss'])

    plt.plot(range(1, len(test_losses)*2, 2), test_losses, label=exp)


plt.legend()
plt.savefig(os.path.join(exps_dir, 'loss.png'), dpi=300, bbox_inches='tight')
plt.show()

plt.close()


plt.figure()

for exp in exp_settings:
    sub_dir = os.path.join(exps_dir, exp)

    test_accs = []

    log_json_files = os.path.join(sub_dir, 'log.json')

    with open(log_json_files, 'r') as f:
        log = json.load(f)
        train_log = log['train']
        test_log = log['test']


    for key, val in test_log.items():
        test_accs.append(val['acc'])

    plt.plot(range(1, len(test_accs)*2, 2), test_accs, label=exp)


plt.legend()
plt.savefig(os.path.join(exps_dir, 'acc.png'), dpi=300, bbox_inches='tight')
plt.show()

