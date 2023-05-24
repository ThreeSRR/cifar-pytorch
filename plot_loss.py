import matplotlib.pyplot as plt
import json
import seaborn as sns
import os

sns.set_theme()


exps_dir = 'experiments/cifar10/0_kernel_size/alexnet'
model = os.path.basename(exps_dir)

for exp in os.listdir(exps_dir):
    sub_dir = os.path.join(exps_dir, exp)

    train_losses = []
    test_losses = []

    train_accs = []
    test_accs = []

    log_json_files = os.path.join(sub_dir, 'log.json')

    with open(log_json_files, 'r') as f:
        log = json.load(f)
        train_log = log['train']
        test_log = log['test']

    for key, val in train_log.items():
        train_losses.append(val['loss'])
        train_accs.append(val['acc'])

    for key, val in test_log.items():
        test_losses.append(val['loss'])
        test_accs.append(val['acc'])


    plt.figure()
    # plt.plot(train_losses, label='Training Loss')
    # plt.plot(range(1, len(test_losses)*2, 2), test_losses, label='Test Loss')

    plt.plot(train_accs, label='Training ACC')
    plt.plot(range(1, len(test_accs)*2, 2), test_accs, label='Test ACC')

    plt.legend()
    # plt.show()

    plt.savefig(os.path.join(sub_dir, '%s-%s-acc.png'%(model, exp)), dpi=300, bbox_inches='tight')


    plt.close()


    plt.figure()
    plt.plot(train_losses, label='Training Loss')
    plt.plot(range(1, len(test_losses)*2, 2), test_losses, label='Test Loss')

    plt.legend()

    plt.savefig(os.path.join(sub_dir, '%s-%s-loss.png'%(model, exp)), dpi=300, bbox_inches='tight')

