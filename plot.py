import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


with open('ex5_10assets/loss_acc.txt', encoding='utf-8') as infile:
    lines = list(infile)
data = {l.split()[0]: list(map(float, l.split()[1:])) for l in lines}
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes[0, 0].plot(data['batch_train_losses'])
axes[0, 0].set_title('train losses')
axes[0, 0].set_ylim(0, max(max(data['batch_train_losses']),
                           max(data['batch_test_losses'])))
axes[0, 0].set_xlabel('batch')
axes[0, 0].set_ylabel('CE-loss')
axes[0, 1].plot(data['batch_train_accuracies'])
axes[0, 1].set_ylim(0, 1)
axes[0, 1].set_title('train accuracies')
axes[0, 1].set_xlabel('batch')
axes[1, 0].plot(data['batch_test_losses'])
axes[1, 0].set_ylim(0, max(max(data['batch_train_losses']),
                           max(data['batch_test_losses'])))
axes[1, 0].set_title('test losses')
axes[1, 0].set_xlabel('batch')
axes[1, 0].set_ylabel('CE-loss')
axes[1, 1].plot(data['batch_test_accuracies'])
axes[1, 1].set_ylim(0, 1)
axes[1, 1].set_title('test accuracies')
axes[1, 1].set_xlabel('batch')
fig.savefig('ex5_10assets/loss_acc.png')
plt.close(fig)
