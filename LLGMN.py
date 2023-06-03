import json
import numpy as np
from matplotlib import pyplot as plt

class LLGMN:
    # パラメータの読み込み
    f = open('parameters.json')
    param = json.load(f)

    # クラス数
    class_num = param['class_num']
    # 入力次元数
    input_dim = param['input_dim']
    # コンポーネント数
    component_num = param['component_num']
    # 学習率
    learning_rate = param['learning_rate']
    # バッチサイズ
    batch_size = param['batch_size']
    # 学習回数
    iters_num = param['iters_num']

    def __init__(self):
        # 変換後の入力次元数
        self.transformed_input_dim = int(1 + LLGMN.input_dim * (LLGMN.input_dim + 3) / 2)
        # 重み
        self.weight = np.random.rand(LLGMN.class_num, LLGMN.component_num, self.transformed_input_dim)
        self.weight[LLGMN.class_num-1, LLGMN.component_num-1, :] = 0
        # 勾配
        self.grad = []

        self.output_shape_checked = False
        self.grad_shape_checked = False

    # 入力の非線形変換
    def input_transformation(self, input_vector):
        transformed_input = []
        transformed_input.append(1)
        transformed_input.extend(input_vector)
        tmp = [input_vector[i] * input_vector[j] for i in range(LLGMN.input_dim) for j in range(i, LLGMN.input_dim)]
        transformed_input.extend(tmp)
        return transformed_input

    def forward(self, batch_data):
        # 第1層
        self.input_1 = np.array([self.input_transformation(input_vector) for input_vector in batch_data])
        self.output_1 = self.input_1
        # 第2層
        self.input_2 = np.array([np.sum(o1 * self.weight, axis=2) for o1 in self.output_1])
        self.output_2 = np.array([np.exp(i2) / np.sum(np.exp(i2)) for i2 in self.input_2])
        # 第3層
        self.input_3 = np.sum(self.output_2, axis=2)
        self.output_3 = self.input_3

        if not self.output_shape_checked:
            print('output1: ', self.output_1.shape)
            print('output2: ', self.output_2.shape)
            print('output3: ', self.output_3.shape)
            self.output_shape_checked = True

    def backward(self, batch_label):
        # 勾配の計算
        tmp = [((self.output_3[i] - batch_label[i]).reshape(LLGMN.class_num, 1)
                * (self.output_2[i] / self.output_3[i].reshape(LLGMN.class_num, 1))).reshape(LLGMN.class_num, LLGMN.component_num, 1)
                * self.input_1[i] for i in range(LLGMN.batch_size)]
        self.grad = np.sum(tmp, axis=0) / LLGMN.batch_size

        if not self.grad_shape_checked:
            print('weight: ', self.weight.shape)
            print('grad: ', self.grad.shape)
            self.grad_shape_checked = True

        # 重みの更新
        self.weight -= LLGMN.learning_rate * self.grad

    def predict(self, data, label):
        self.forward(data)
        # 正解率（平均）
        acc = sum(np.argmax(label_vector) == np.argmax(output_vector) for label_vector, output_vector in zip(label, self.output_3)) / len(data)
        # 交差エントロピー（平均）
        loss =  -np.sum(label * np.log(self.output_3 + 1e-7)) / len(data)
        return acc, loss

    def train(self, train_data, train_label, test_data, test_label):
        # 正解率
        train_acc_list = []
        test_acc_list = []
        # 交差エントロピー
        train_loss_list = []
        test_loss_list = []

        epoch_list = []

        iter_per_epoch = max(int(len(train_data) / LLGMN.batch_size), 1)

        for i in range(LLGMN.iters_num):
            batch_mask = np.random.choice(len(train_data), size=LLGMN.batch_size, replace=False)
            self.forward(train_data[batch_mask])
            self.backward(train_label[batch_mask])

            if i % iter_per_epoch == 0:
                train_acc, train_loss = self.predict(train_data, train_label)
                test_acc, test_loss = self.predict(test_data, test_label)
                train_acc_list.append(train_acc)
                train_loss_list.append(train_loss)
                test_acc_list.append(test_acc)
                test_loss_list.append(test_loss)
                epoch_list.append(i / iter_per_epoch)
                print(f'[epoch {(i / iter_per_epoch):.0f}] train_acc: {train_acc:.3f}, train_loss: {train_loss:.3f}, test_acc: {test_acc:.3f}, test_loss: {test_loss:.3f}')

        print('training finished')
        self.plot_line(epoch_list, train_acc_list, test_acc_list, 'accuracy')
        self.plot_line(epoch_list, train_loss_list, test_loss_list, 'loss')

    # グラフ描画
    def plot_line(self, x, y1, y2, label_name):
        fig, ax = plt.subplots()
        ax.set_xlabel('epoch')
        ax.set_ylabel(label_name)
        ax.plot(x, y1, color='tab:blue', label='train')
        ax.plot(x, y2, color='tab:orange', label='test')
        ax.legend()
        fig.tight_layout()
        plt.show()

if __name__ == "__main__":
    # 筋電データの読み込み
    train_data = np.loadtxt('data/train_data.txt', delimiter='\t', skiprows=1)
    train_label = np.loadtxt('data/train_label.txt', delimiter='\t', skiprows=1)
    test_data = np.loadtxt('data/test_data.txt', delimiter='\t', skiprows=1)
    test_label = np.loadtxt('data/test_label.txt', delimiter='\t', skiprows=1)
    llgmn = LLGMN()
    llgmn.train(train_data, train_label, test_data, test_label)