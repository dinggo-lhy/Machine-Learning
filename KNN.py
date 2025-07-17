import numpy as np  #  这个是用于计算的
import matplotlib   #  进行绘图以及可视化
matplotlib.use('TKAgg')  #  防止报错添加的
import matplotlib.pyplot as plt   #绘图+可视化
plt.rcParams['font.sans-serif'] = [
    'Microsoft YaHei',  # 微软雅黑（Windows）
]#这个是因为画图时中文会变成方框，但是使用系统字体之后就可以显示了
from sklearn.metrics import roc_curve, auc#这个是评价函数，ROC，AUC的
import pandas as pd#读取数据+数据处理用的
from sklearn.preprocessing import label_binarize#标签二值化，进行多分类ROC绘制
from sklearn.metrics import confusion_matrix, classification_report#混淆矩阵以及生成分类报告
import time#这个就是计算训练时长的
import seaborn as sns#可视化，绘制混淆矩阵图片的
# ====================== 模型保存模块 ======================
import pickle

def 保存模型(模型对象, 文件路径):
    """通用模型保存方法"""
    try:
        with open(文件路径, 'wb') as f:
            pickle.dump(模型对象, f)
        print(f"模型已保存至: {文件路径}")
    except Exception as e:
        print(f"保存失败: {str(e)}")
# ====================== 基础工具函数 ======================
def 划分数据集(特征集, 标签集, 测试集比例=0.3):
    """随机划分训练集和测试集"""
    总样本数 = len(特征集)
    索引列表 = list(range(总样本数))
    np.random.shuffle(索引列表)#这里先打乱顺序，防止训练集和测试集不均衡
    切分点 = int(总样本数 * (1 - 测试集比例))

    训练特征 = [特征集[i] for i in 索引列表[:切分点]]
    训练标签 = [标签集[i] for i in 索引列表[:切分点]]
    测试特征 = [特征集[i] for i in 索引列表[切分点:]]
    测试标签 = [标签集[i] for i in 索引列表[切分点:]]

    return np.array(训练特征), np.array(训练标签), np.array(测试特征), np.array(测试标签)


def 标准化数据(特征集):
    """Z-score标准化"""
    均值 = np.mean(特征集, axis=0)
    标准差 = np.std(特征集, axis=0)
    return (特征集 - 均值) / (标准差 + 1e-8), 均值, 标准差#数值稳定性技巧（防止除0错误）


def 应用标准化(特征集, 均值, 标准差):#对新数据进行同样的处理，和训练数据变化必须一样
    """应用预计算的标准化参数"""
    return (特征集 - 均值) / (标准差 + 1e-8)


def 二值化标签(标签集, 当前类别):#1对其他的二分类
    """将多分类标签转换为二分类（One-vs-Rest）"""
    return np.array([1 if 标签 == 当前类别 else 0 for 标签 in 标签集])


# ====================== 自实现算法 ======================
class 贝叶斯决策器:#假设特征之间条件独立，好算，好写
    """朴素贝叶斯分类器"""

    def __init__(self):
        self.类别先验 = None
        self.类别均值 = None
        self.类别方差 = None

    def 训练(self, 特征集, 标签集):
        类别列表 = np.unique(标签集)#先获得类别有哪些
        self.类别列表 = 类别列表
        self.类别先验 = []
        self.类别均值 = []
        self.类别方差 = []

        for 类别 in 类别列表:
            类别索引 = np.where(标签集 == 类别)[0]
            self.类别先验.append(len(类别索引) / len(标签集))

            # 计算均值和方差
            类别特征 = 特征集[类别索引]
            self.类别均值.append(np.mean(类别特征, axis=0))
            self.类别方差.append(np.var(类别特征, axis=0) + 1e-8)  # 避免零方差
            #计算均值，方差

    def 预测概率(self, 样本):#就是计算出概率，然后判断是哪一个
        对数概率列表 = []  # 改用对数概率避免数值下溢
        for 先验, 均值, 方差 in zip(self.类别先验, self.类别均值, self.类别方差):
            # 确保方差不为零
            方差 = np.clip(方差, 1e-8, None)

            # 计算对数似然 (更稳定的计算方式)
            对数归一化项 = -0.5 * np.sum(np.log(2 * np.pi * 方差))
            指数项 = -0.5 * np.sum((样本 - 均值) ** 2 / 方差)
            对数似然 = 对数归一化项 + 指数项

            # 对数后验概率 = log(先验) + log(似然)
            对数概率列表.append(np.log(先验) + 对数似然)

        # 使用log-sum-exp技巧避免数值溢出
        最大对数概率 = np.max(对数概率列表)
        对数概率列表 = 对数概率列表 - 最大对数概率
        概率列表 = np.exp(对数概率列表)

        # 归一化
        总概率 = np.sum(概率列表)
        return 概率列表 / (总概率 + 1e-10)  # 防止除以零

class Fisher线性判别器:#有监督的降为，主要需要求类内离散度矩阵，类间离散度矩阵，投影方向
    """Fisher线性判别分析"""

    def __init__(self):
        self.投影矩阵 = None
        self.类别中心 = None

    def 训练(self, 特征集, 标签集):
        类别列表 = np.unique(标签集)
        总均值 = np.mean(特征集, axis=0)

        # 计算类内离散度矩阵
        类内散布 = np.zeros((特征集.shape[1], 特征集.shape[1]))
        for 类别 in 类别列表:
            类别索引 = np.where(标签集 == 类别)[0]
            类别特征 = 特征集[类别索引]
            类别均值 = np.mean(类别特征, axis=0)
            类内散布 += (类别特征 - 类别均值).T @ (类别特征 - 类别均值)

        # 计算类间离散度矩阵
        类间散布 = np.zeros((特征集.shape[1], 特征集.shape[1]))
        for 类别 in 类别列表:
            类别索引 = np.where(标签集 == 类别)[0]
            类别均值 = np.mean(特征集[类别索引], axis=0)
            n = len(类别索引)
            类间散布 += n * (类别均值 - 总均值).reshape(-1, 1) @ (类别均值 - 总均值).reshape(1, -1)

        # 求解广义特征值问题
        特征值, 特征向量 = np.linalg.eig(np.linalg.inv(类内散布) @ 类间散布)
        排序索引 = np.argsort(特征值)[::-1]
        self.投影矩阵 = 特征向量[:, 排序索引[:len(类别列表) - 1]]

    def 投影(self, 特征集):
        return 特征集 @ self.投影矩阵


class K近邻分类器:#分类结果由最近邻的k个样本决定
    """K最近邻分类器"""

    def __init__(self, k值=5):
        self.k值 = k值
        self.训练特征 = None
        self.训练标签 = None

    def 训练(self, 特征集, 标签集):
        self.训练特征 = 特征集
        self.训练标签 = 标签集

    def 预测概率(self, 样本集):#主要就是计算距离，使用的欧式距离进行判断，关键在于查找最近邻和计算类别概率
        概率列表 = []
        for 样本 in 样本集:
            # 计算距离
            距离 = np.sqrt(np.sum((self.训练特征 - 样本) ** 2, axis=1))
            # 获取最近的k个样本
            最近索引 = np.argsort(距离)[:self.k值]
            最近标签 = self.训练标签[最近索引]

            # 计算各类别概率
            类别概率 = []
            for 类别 in np.unique(self.训练标签):
                类别计数 = np.sum(最近标签 == 类别)
                类别概率.append(类别计数 / self.k值)
            概率列表.append(类别概率)
        return np.array(概率列表)


# ====================== 多分类ROC评估 ======================
def 计算多分类ROC(真实标签, 预测概率):#计算多分类问题中各类别的ROC曲线，标签二值化，逐类ROC计算​​，Micro平均计算计算全局FPR和TPR，Macro平均计算计算平均TPR曲线和AUC
    """计算多分类ROC曲线，支持micro和macro平均"""
    # 二值化真实标签
    n_classes = 预测概率.shape[1]
    y_true_bin = label_binarize(真实标签, classes=np.arange(n_classes))

    # 计算每个类别的ROC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], 预测概率[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # 计算micro平均ROC
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), 预测概率.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # 计算macro平均ROC
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    return fpr, tpr, roc_auc


def 绘制ROC曲线(fpr, tpr, roc_auc, 分类器名称):#就是一个划线算法
    """绘制包含micro和macro平均的ROC曲线"""
    plt.figure(figsize=(10, 8))
    颜色 = plt.cm.rainbow(np.linspace(0, 1, len(roc_auc) - 2))  # 减去micro和macro

    # 绘制每个类别的ROC曲线
    for i, 颜色 in enumerate(颜色):
        plt.plot(fpr[i], tpr[i], color=颜色, lw=1.5, alpha=0.6,
                 label=f'类别 {i} (AUC={roc_auc[i]:.2f})')

    # 绘制平均ROC曲线
    plt.plot(fpr["micro"], tpr["micro"], color='deeppink', linestyle=':', lw=3,
             label=f'Micro平均 (AUC={roc_auc["micro"]:.2f})')
    plt.plot(fpr["macro"], tpr["macro"], color='navy', linestyle=':', lw=3,
             label=f'Macro平均 (AUC={roc_auc["macro"]:.2f})')

    # 绘制随机猜测线
    plt.plot([0, 1], [0, 1], 'k--', lw=1.5)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假阳性率 (FPR)', fontsize=12)
    plt.ylabel('真阳性率 (TPR)', fontsize=12)
    plt.title(f'{分类器名称} - 多分类ROC曲线', fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.show()

# ====================== 添加准确率计算函数 ======================
def 计算准确率(真实标签, 预测概率):
    """计算分类器准确率"""
    # 获取预测标签（选择概率最高的类别）
    预测标签 = np.argmax(预测概率, axis=1)
    # 计算准确率
    正确数 = np.sum(预测标签 == 真实标签)
    总样本数 = len(真实标签)
    return 正确数 / 总样本数


def 输出详细评估(真实标签, 预测概率, 类别名称, 分类器名称):#控制台输出分类报告，绘制混淆矩阵图
    """输出混淆矩阵和分类报告"""
    # 获取预测标签（概率最大值对应类别）
    预测标签 = np.argmax(预测概率, axis=1)

    # 1. 打印分类报告（包含精确率/召回率/F1等）
    print(f"\n【{分类器名称}分类报告】")
    print(classification_report(真实标签, 预测标签, target_names=类别名称, digits=4))

    # 2. 绘制混淆矩阵热力图
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(真实标签, 预测标签)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=类别名称,
                yticklabels=类别名称)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title(f'{分类器名称} - 混淆矩阵')
    plt.show()
# ====================== 主程序 ======================
if __name__ == "__main__":
    # 1. 从Excel读取数据集，我有两个Excel表格，一个D:\应用数据\onedive\OneDrive\Desktop\1\DryBeanDataset\DryBeanDataset\Dry_Bean_Dataset.xlsx，
    # 另一个D:\应用数据\onedive\OneDrive\Desktop\1\processed_features.xlsx，这个是特征提取与选择之后的。
    # 特征提取与选择之后的数据集原本是CSV格式，直接改后缀仍然报错，原因是xlsx是一个压缩的形式，我使用WPS以xlsx格式保存后正常了
    文件路径 = r'D:\应用数据\onedive\OneDrive\Desktop\1\processed_features.xlsx'
    数据框 = pd.read_excel(文件路径, engine='openpyxl')#制定打开xlsx格式文件

    # 分离特征和标签
    特征集 = 数据框.iloc[:, :-1].values.astype(float)  # 所有列除最后一列
    标签集 = 数据框.iloc[:, -1].values  # 最后一列是类别

    # 将字符串标签编码为整数，方便计算处理
    类别列表, 标签集 = np.unique(标签集, return_inverse=True)
    print(f"加载完成，共 {len(特征集)} 个样本，{len(类别列表)} 个类别")
    # 2. 数据预处理
    特征集, 均值, 标准差 = 标准化数据(特征集)
    训练特征, 训练标签, 测试特征, 测试标签 = 划分数据集(特征集, 标签集, 测试集比例=0.3)
    # 3. 训练不同分类器
    print("训练贝叶斯分类器...")
    开始时间 = time.time()
    贝叶斯模型 = 贝叶斯决策器()
    贝叶斯模型.训练(训练特征, 训练标签)
    贝叶斯概率 = np.array([贝叶斯模型.预测概率(样本) for 样本 in 测试特征])
    结束时间 = time.time()
    print(f"训练时间: {结束时间 - 开始时间:.2f}秒")
    # 计算并输出贝叶斯准确率
    贝叶斯准确率 = 计算准确率(测试标签, 贝叶斯概率)
    print(f"贝叶斯分类器准确率: {贝叶斯准确率:.4f}")
    print("训练Fisher判别器...")
    开始时间 = time.time()
    fisher模型 = Fisher线性判别器()
    fisher模型.训练(训练特征, 训练标签)
    fisher特征 = fisher模型.投影(测试特征)
    # 使用KNN在Fisher投影空间中进行分类
    fisher_knn = K近邻分类器(k值=5)
    fisher_knn.训练(fisher模型.投影(训练特征), 训练标签)
    fisher概率 = fisher_knn.预测概率(fisher特征)
    结束时间 = time.time()
    print(f"训练时间: {结束时间 - 开始时间:.2f}秒")
    # 计算并输出Fisher准确率
    fisher准确率 = 计算准确率(测试标签, fisher概率)
    print(f"Fisher+KNN分类器准确率: {fisher准确率:.4f}")
    print("训练K近邻分类器...")
    开始时间 = time.time()
    knn模型 = K近邻分类器(k值=5)
    knn模型.训练(训练特征, 训练标签)
    knn概率 = knn模型.预测概率(测试特征)
    结束时间 = time.time()
    print(f"训练时间: {结束时间 - 开始时间:.2f}秒")
    # 计算并输出KNN准确率
    knn准确率 = 计算准确率(测试标签, knn概率)
    print(f"K近邻分类器准确率: {knn准确率:.4f}")
    # 4. 评估并绘制ROC曲线
    类别名称 = [f"类别{i}" for i in range(len(类别列表))]  # 提前定义类别名称

    print(f"准确率: {贝叶斯准确率:.4f} | ", end="")  # 显示准确率 [新增]
    贝叶斯_fpr, 贝叶斯_tpr, 贝叶斯_auc = 计算多分类ROC(测试标签, 贝叶斯概率)
    绘制ROC曲线(贝叶斯_fpr, 贝叶斯_tpr, 贝叶斯_auc, "贝叶斯分类器")
    输出详细评估(测试标签, 贝叶斯概率, 类别名称, "贝叶斯")  # 新增详细评估

    print(f"准确率: {fisher准确率:.4f} | ", end="")  # 显示准确率 [新增]
    fisher_fpr, fisher_tpr, fisher_auc = 计算多分类ROC(测试标签, fisher概率)
    绘制ROC曲线(fisher_fpr, fisher_tpr, fisher_auc, "Fisher+KNN分类器")
    输出详细评估(测试标签, fisher概率, 类别名称, "Fisher+KNN")  # 新增详细评估

    print(f"准确率: {knn准确率:.4f} | ", end="")  # 显示准确率 [新增]
    knn_fpr, knn_tpr, knn_auc = 计算多分类ROC(测试标签, knn概率)
    绘制ROC曲线(knn_fpr, knn_tpr, knn_auc, "K近邻分类器")
    输出详细评估(测试标签, knn概率, 类别名称, "K近邻")  # 新增详细评估

    print("\n正在保存模型...")
    # 保存贝叶斯模型
    保存模型(贝叶斯模型, 'bayes_model.pkl')
    # 保存Fisher投影模型
    保存模型(fisher模型, 'fisher_model.pkl')
    # 保存Fisher空间KNN分类器
    保存模型(fisher_knn, 'fisher_knn_model.pkl')
    # 保存KNN模型
    保存模型(knn模型, 'knn_model.pkl')
    # 保存标准化参数
    with open('scaler_params.pkl', 'wb') as f:
        pickle.dump({'mean': 均值, 'std': 标准差}, f)

