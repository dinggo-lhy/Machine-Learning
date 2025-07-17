import numpy as np
import matplotlib

matplotlib.use('TKAgg')  # 非交互式后端（适合生成图片文件）
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = [
    'Microsoft YaHei',  # 微软雅黑（Windows）
]
# 解决负号显示方块问题
plt.rcParams['axes.unicode_minus'] = False
from sklearn.metrics import roc_curve, auc
import pandas as pd
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix, classification_report
import time
import seaborn as sns
import joblib
from joblib import Parallel, delayed
from math import sqrt
import warnings

warnings.filterwarnings('ignore')

# ====================== 模型保存模块 ======================
import pickle


def 保存模型(模型对象, 文件路径):
    """通用模型保存方法"""
    try:
        with open(文件路径, 'wb') as f:
            pickle.dump(模型对象, f)
        print(f"✅ 模型已保存至: {文件路径}")
    except Exception as e:
        print(f"❌ 保存失败: {str(e)}")


# ====================== 基础工具函数 ======================
def 划分数据集(特征集, 标签集, 测试集比例=0.3):
    """随机划分训练集和测试集"""
    总样本数 = len(特征集)
    索引列表 = list(range(总样本数))
    np.random.shuffle(索引列表)
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
    return (特征集 - 均值) / (标准差 + 1e-8), 均值, 标准差


def 应用标准化(特征集, 均值, 标准差):
    """应用预计算的标准化参数"""
    return (特征集 - 均值) / (标准差 + 1e-8)


def 二值化标签(标签集, 当前类别):
    """将多分类标签转换为二分类（One-vs-Rest）"""
    return np.array([1 if 标签 == 当前类别 else 0 for 标签 in 标签集])


# ====================== 自实现算法 ======================
class 贝叶斯决策器:
    """朴素贝叶斯分类器"""

    def __init__(self):
        self.类别先验 = None
        self.类别均值 = None
        self.类别方差 = None

    def 训练(self, 特征集, 标签集):
        类别列表 = np.unique(标签集)
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

    def 预测概率(self, 样本):
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


class Fisher线性判别器:
    """Fisher线性判别分析"""

    def __init__(self):
        self.投影矩阵 = None
        self.类别中心 = None

    def 训练(self, 特征集, 标签集):
        类别列表 = np.unique(标签集)
        总均值 = np.mean(特征集, axis=0)

        # 计算类内散布矩阵
        类内散布 = np.zeros((特征集.shape[1], 特征集.shape[1]))
        for 类别 in 类别列表:
            类别索引 = np.where(标签集 == 类别)[0]
            类别特征 = 特征集[类别索引]
            类别均值 = np.mean(类别特征, axis=0)
            类内散布 += (类别特征 - 类别均值).T @ (类别特征 - 类别均值)

        # 计算类间散布矩阵
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


class K近邻分类器:
    """K最近邻分类器"""

    def __init__(self, k值=5):
        self.k值 = k值
        self.训练特征 = None
        self.训练标签 = None

    def 训练(self, 特征集, 标签集):
        self.训练特征 = 特征集
        self.训练标签 = 标签集

    def 预测概率(self, 样本集):
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


# ====================== 决策树分类器 ======================
class 决策树分类器:
    """基于基尼系数的决策树分类器"""
    #最大深度​：控制树复杂度，防止过拟合
    #最小分裂样本​：避免在样本过少的节点上分裂
    #最小叶节点样本​：确保叶节点有统计意义

    def __init__(self, 最大深度=None, 最小分裂样本=10, 最小叶节点样本=5):
        self.最大深度 = 最大深度
        self.最小分裂样本 = 最小分裂样本
        self.最小叶节点样本 = 最小叶节点样本
        self.树结构 = None
        self.特征重要性 = None
        self.类别数 = None

    def 训练(self, 特征集, 标签集):#记录特征数量和类别数量，初始化特征重要性数组（全零），调用_生长树方法从根节点开始构建决策树
        self.特征数 = 特征集.shape[1]
        self.类别数 = len(np.unique(标签集))
        self.特征重要性 = np.zeros(self.特征数)
        self.树结构 = self._生长树(特征集, 标签集, 深度=0)

    def _生长树(self, 特征集, 标签集, 深度):#
        n_samples = len(标签集)

        # 终止条件检查：1.达到最大深度，2.当前节点标签完全相同（纯节点），3.节点样本数小于最小分裂样本，完成之后创建叶节点
        if (self.最大深度 is not None and 深度 >= self.最大深度) or \
                len(np.unique(标签集)) == 1 or \
                n_samples < self.最小分裂样本:
            return self._创建叶节点(标签集)

        # 寻找最佳分裂：1.尝试所有候选特征和分割点，2.选择最大基尼增益的分裂
        最佳特征, 最佳分割值 = self._寻找最佳分裂(特征集, 标签集)

        if 最佳特征 is None:
            return self._创建叶节点(标签集)

        # 分裂数据集
        左掩码 = 特征集[:, 最佳特征] <= 最佳分割值
        右掩码 = ~左掩码

        # 分裂验证​​：检查分裂后的子节点是否满足最小叶节点样本要求，如果不满足则停止分裂
        if np.sum(左掩码) < self.最小叶节点样本 or np.sum(右掩码) < self.最小叶节点样本:
            return self._创建叶节点(标签集)

        # 递归构建子树：对左子树和右子树递归调用生长函数
        左子树 = self._生长树(特征集[左掩码], 标签集[左掩码], 深度 + 1)
        右子树 = self._生长树(特征集[右掩码], 标签集[右掩码], 深度 + 1)

        return {
            '特征索引': 最佳特征,
            '分割值': 最佳分割值,
            '左': 左子树,
            '右': 右子树
        }

    def _寻找最佳分裂(self, 特征集, 标签集):
        最佳增益 = 0.01  # 增益小于此值不分裂
        最佳特征 = None
        最佳分割值 = None

        # 随机选择特征子集(√n个随机特征)，减少计算量，增加随机性
        候选特征数 = int(sqrt(self.特征数))
        特征索引 = np.random.choice(self.特征数, 候选特征数, replace=False)

        for 特征 in 特征索引:
            特征值 = 特征集[:, 特征]

            # 优化点：离散特征：使用所有唯一值，显著减少计算复杂度
            if len(np.unique(特征值)) > 20:
                候选分割点 = np.percentile(特征值, np.linspace(10, 90, 10))
            else:
                候选分割点 = np.unique(特征值)

            for 分割值 in 候选分割点:
                增益 = self._计算基尼增益(标签集, 特征值, 分割值)
                if 增益 > 最佳增益:
                    最佳增益 = 增益
                    最佳特征 = 特征
                    最佳分割值 = 分割值

        # 更新特征重要性：最佳特征的增益累加到重要性数组
        if 最佳特征 is not None:
            self.特征重要性[最佳特征] += 最佳增益

        return 最佳特征, 最佳分割值

    def _计算基尼增益(self, 标签集, 特征值, 分割值):#使用公式进行计算
        """计算基尼不纯度增益"""
        左掩码 = 特征值 <= 分割值
        右掩码 = ~左掩码

        n_left = np.sum(左掩码)
        n_right = np.sum(右掩码)
        n_total = len(标签集)

        if n_left == 0 or n_right == 0:
            return 0

        # 计算父节点的基尼不纯度
        父基尼 = self._计算基尼(标签集)

        # 计算子节点的基尼不纯度
        左基尼 = self._计算基尼(标签集[左掩码])
        右基尼 = self._计算基尼(标签集[右掩码])

        # 计算加权平均基尼不纯度
        加权基尼 = (n_left / n_total) * 左基尼 + (n_right / n_total) * 右基尼

        # 返回基尼增益
        return 父基尼 - 加权基尼

    def _计算基尼(self, 标签集):
        """计算基尼不纯度"""
        _, 计数 = np.unique(标签集, return_counts=True)
        概率 = 计数 / len(标签集)
        return 1 - np.sum(概率 ** 2)

    def _创建叶节点(self, 标签集):
        """创建叶子节点，存储类别分布"""#每个叶节点包含类别概率向量，概率向量长度等于类别数
        # 创建全零向量（长度=总类别数）
        概率向量 = np.zeros(self.类别数)

        # 计算当前节点的类别分布
        if len(标签集) > 0:
            类别, 计数 = np.unique(标签集, return_counts=True)
            类别概率 = 计数 / len(标签集)

            # 将概率分配到对应位置
            for 类别索引, 概率值 in zip(类别, 类别概率):
                概率向量[类别索引] = 概率值

        return {'叶节点': True, '概率': 概率向量}

    def 预测概率(self, 特征集):
        """预测每个样本属于各个类别的概率"""#1.从根节点开始​​：遍历树结构
        # 2.路径决策​​：根据当前节点的分裂规则向左/右子树移动
        # 3.到达叶节点​​：返回叶节点的概率向量
        # 4.批量预测​​：循环处理每个样本
        n_samples = 特征集.shape[0]
        概率 = np.zeros((n_samples, self.类别数))

        for i in range(n_samples):
            节点 = self.树结构
            while not 节点.get('叶节点', False):
                if 特征集[i, 节点['特征索引']] <= 节点['分割值']:
                    节点 = 节点['左']
                else:
                    节点 = 节点['右']
            概率[i] = 节点['概率']
        return 概率


# ====================== 随机森林分类器 ======================
class 随机森林分类器:
    """基于决策树的随机森林分类器"""

    def __init__(self, 树数量=1000, 最大深度=8, 最小分裂样本=10,
                 最小叶节点样本=5, 并行数=-1, 随机种子=None):#类定义和初始化
        self.树数量 = 树数量
        self.最大深度 = 最大深度
        self.最小分裂样本 = 最小分裂样本
        self.最小叶节点样本 = 最小叶节点样本
        self.并行数 = 并行数  # 并行训练树的数量
        self.随机种子 = 随机种子
        self.决策树列表 = []
        self.特征重要性 = None
        self.类别列表 = None
        self.类别数 = None

        if 随机种子 is not None:
            np.random.seed(随机种子)

    def 训练(self, 特征集, 标签集):#训练方法：获取样本数和特征数，提取唯一类别标签，初始化特征重要性数组

        n_samples, n_features = 特征集.shape
        self.类别列表 = np.unique(标签集)
        self.类别数 = len(self.类别列表)
        self.特征重要性 = np.zeros(n_features)

        # 并行训练决策树，使用joblib.Parallel并行训练多棵树，delayed函数确保每棵树独立训练，每棵树通过_训练单棵树方法创建
        self.决策树列表 = Parallel(n_jobs=self.并行数)(
            delayed(self._训练单棵树)(特征集, 标签集, i)
            for i in range(self.树数量)
        )

        # 计算特征重要性：累积所有树的特征重要性，取平均值得到全局特征重要性
        for 树 in self.决策树列表:
            self.特征重要性 += 树.特征重要性
        self.特征重要性 /= self.树数量

    def _训练单棵树(self, 特征集, 标签集, 树索引):
        # 1. 自助采样（有放回）
        n_samples = len(标签集)
        索引 = np.random.choice(n_samples, n_samples, replace=True)#约63.2%的原始样本被选中，剩余36.8%形成Out-of-Bag样本，可用于验证
        特征子集 = 特征集[索引]
        标签子集 = 标签集[索引]

        # 2. 创建并训练决策树
        树 = 决策树分类器(
            最大深度=self.最大深度,
            最小分裂样本=self.最小分裂样本,
            最小叶节点样本=self.最小叶节点样本
        )
        树.训练(特征子集, 标签子集)
        return 树

    def 预测概率(self, 特征集):
        """预测每个类别的概率"""
        # 并行预测每棵树的概率，每棵树独立预测类别概率，利用并行处理加速计算
        所有预测 = Parallel(n_jobs=self.并行数)(
            delayed(树.预测概率)(特征集)
            for 树 in self.决策树列表
        )

        # 平均所有树的预测概率
        return np.mean(所有预测, axis=0)

    def 预测(self, 特征集):#选择概率最高的类别作为最终预测
        """预测类别标签"""
        概率 = self.预测概率(特征集)
        return np.argmax(概率, axis=1)

    def 可视化特征重要性(self, 特征名称):#通过图表输出特征重要性
        """可视化特征重要性"""
        if self.特征重要性 is None:
            print("请先训练模型")
            return

        # 排序特征重要性
        排序索引 = np.argsort(self.特征重要性)[::-1]
        排序重要性 = self.特征重要性[排序索引]
        排序名称 = [特征名称[i] for i in 排序索引]

        # 创建图表
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(排序名称)), 排序重要性, align='center')
        plt.yticks(range(len(排序名称)), 排序名称)
        plt.xlabel('特征重要性', fontsize=12)
        plt.title('随机森林特征重要性', fontsize=14)
        plt.gca().invert_yaxis()  # 最重要特征在顶部
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig('feature_importance.png')  # 保存图片防止窗口关闭
        plt.show()


# ====================== 多分类ROC评估 ======================
def 计算多分类ROC(真实标签, 预测概率):
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


def 绘制ROC曲线(fpr, tpr, roc_auc, 分类器名称):
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
    plt.savefig(f'{分类器名称}_ROC.png')  # 保存图片
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


def 输出详细评估(真实标签, 预测概率, 类别名称, 分类器名称):
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
    plt.savefig(f'{分类器名称}_混淆矩阵.png')  # 保存图片
    plt.show()


# ====================== 主程序 ======================
if __name__ == "__main__":
    # ========== 替换开始 ==========
    # 1. 从Excel读取数据集
    文件路径 = r'D:\应用数据\onedive\OneDrive\Desktop\1\processed_features.xlsx'
    数据框 = pd.read_excel(文件路径, engine='openpyxl')

    # 分离特征和标签
    特征集 = 数据框.iloc[:, :-1].values.astype(float)  # 所有列除最后一列
    标签集 = 数据框.iloc[:, -1].values  # 最后一列是类别

    # 将字符串标签编码为整数
    类别列表, 标签集 = np.unique(标签集, return_inverse=True)
    print(f"加载完成，共 {len(特征集)} 个样本，{len(类别列表)} 个类别")
    # ========== 替换结束 ==========

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
    fisher准确率 = 计算准确率(测试标签, fisher概率)
    print(f"Fisher+KNN分类器准确率: {fisher准确率:.4f}")

    print("训练K近邻分类器...")
    开始时间 = time.time()
    knn模型 = K近邻分类器(k值=5)
    knn模型.训练(训练特征, 训练标签)
    knn概率 = knn模型.预测概率(测试特征)
    结束时间 = time.time()
    print(f"训练时间: {结束时间 - 开始时间:.2f}秒")
    knn准确率 = 计算准确率(测试标签, knn概率)
    print(f"K近邻分类器准确率: {knn准确率:.4f}")

    # ========== 决策树和随机森林 ==========
    print("训练决策树分类器...")
    开始时间 = time.time()
    决策树模型 = 决策树分类器(最大深度=8, 最小分裂样本=10, 最小叶节点样本=5)
    决策树模型.训练(训练特征, 训练标签)
    决策树概率 = 决策树模型.预测概率(测试特征)
    结束时间 = time.time()
    print(f"训练时间: {结束时间 - 开始时间:.2f}秒")
    决策树准确率 = 计算准确率(测试标签, 决策树概率)
    print(f"决策树分类器准确率: {决策树准确率:.4f}")

    print("训练随机森林分类器(并行加速)...")
    开始时间 = time.time()
    随机森林模型 = 随机森林分类器(
        树数量=1000,
        最大深度=8,
        最小分裂样本=10,
        最小叶节点样本=5,
        并行数=-1,  # 使用所有CPU核心
        随机种子=42
    )
    随机森林模型.训练(训练特征, 训练标签)
    随机森林概率 = 随机森林模型.预测概率(测试特征)
    结束时间 = time.time()
    print(f"训练时间: {结束时间 - 开始时间:.2f}秒")
    随机森林准确率 = 计算准确率(测试标签, 随机森林概率)
    print(f"随机森林分类器准确率: {随机森林准确率:.4f}")

    # 4. 评估并绘制ROC曲线
    类别名称 = [f"类别{i}" for i in range(len(类别列表))]  # 提前定义类别名称

    # 贝叶斯评估
    print(f"准确率: {贝叶斯准确率:.4f} | ", end="")
    贝叶斯_fpr, 贝叶斯_tpr, 贝叶斯_auc = 计算多分类ROC(测试标签, 贝叶斯概率)
    绘制ROC曲线(贝叶斯_fpr, 贝叶斯_tpr, 贝叶斯_auc, "贝叶斯分类器")
    输出详细评估(测试标签, 贝叶斯概率, 类别名称, "贝叶斯")

    # Fisher评估
    print(f"准确率: {fisher准确率:.4f} | ", end="")
    fisher_fpr, fisher_tpr, fisher_auc = 计算多分类ROC(测试标签, fisher概率)
    绘制ROC曲线(fisher_fpr, fisher_tpr, fisher_auc, "Fisher+KNN分类器")
    输出详细评估(测试标签, fisher概率, 类别名称, "Fisher+KNN")

    # KNN评估
    print(f"准确率: {knn准确率:.4f} | ", end="")
    knn_fpr, knn_tpr, knn_auc = 计算多分类ROC(测试标签, knn概率)
    绘制ROC曲线(knn_fpr, knn_tpr, knn_auc, "K近邻分类器")
    输出详细评估(测试标签, knn概率, 类别名称, "K近邻")

    # 决策树评估
    print(f"准确率: {决策树准确率:.4f} | ", end="")
    决策树_fpr, 决策树_tpr, 决策树_auc = 计算多分类ROC(测试标签, 决策树概率)
    绘制ROC曲线(决策树_fpr, 决策树_tpr, 决策树_auc, "决策树分类器")
    输出详细评估(测试标签, 决策树概率, 类别名称, "决策树")

    # 随机森林评估
    print(f"准确率: {随机森林准确率:.4f} | ", end="")
    随机森林_fpr, 随机森林_tpr, 随机森林_auc = 计算多分类ROC(测试标签, 随机森林概率)
    绘制ROC曲线(随机森林_fpr, 随机森林_tpr, 随机森林_auc, "随机森林分类器")
    输出详细评估(测试标签, 随机森林概率, 类别名称, "随机森林")

    # 可视化随机森林特征重要性
    特征名称 = 数据框.columns[:-1].tolist()  # 获取特征名称
    随机森林模型.可视化特征重要性(特征名称)

    # 5. 保存所有模型
    print("\n正在保存模型...")
    # 保存贝叶斯模型
    保存模型(贝叶斯模型, 'bayes_model.pkl')
    # 保存Fisher投影模型
    保存模型(fisher模型, 'fisher_model.pkl')
    # 保存Fisher空间KNN分类器
    保存模型(fisher_knn, 'fisher_knn_model.pkl')
    # 保存KNN模型
    保存模型(knn模型, 'knn_model.pkl')
    # 保存决策树模型
    保存模型(决策树模型, 'decision_tree_model.pkl')
    # 保存随机森林模型
    保存模型(随机森林模型, 'random_forest_model.pkl')
    # 保存标准化参数
    with open('scaler_params.pkl', 'wb') as f:
        pickle.dump({'mean': 均值, 'std': 标准差}, f)

    print("✅ 所有任务完成!")