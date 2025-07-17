import numpy as np
import pandas as pd
import pickle
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


# ====================== 核心模型定义 ======================
class 贝叶斯决策器:
    """朴素贝叶斯分类器实现"""

    def __init__(self):
        self.类别先验 = None
        self.类别均值 = None
        self.类别方差 = None
        self.类别列表 = None

    def 训练(self, 特征集, 标签集):
        self.类别列表 = np.unique(标签集)
        self.类别先验 = []
        self.类别均值 = []
        self.类别方差 = []

        for 类别 in self.类别列表:
            类别索引 = np.where(标签集 == 类别)[0]
            self.类别先验.append(len(类别索引) / len(标签集))
            类别特征 = 特征集[类别索引]
            self.类别均值.append(np.mean(类别特征, axis=0))
            self.类别方差.append(np.var(类别特征, axis=0) + 1e-8)  # 避免零方差

    def 预测概率(self, 样本):
        对数概率列表 = []
        for 先验, 均值, 方差 in zip(self.类别先验, self.类别均值, self.类别方差):
            方差 = np.clip(方差, 1e-8, None)  # 确保方差不为零
            对数归一化项 = -0.5 * np.sum(np.log(2 * np.pi * 方差))
            指数项 = -0.5 * np.sum((样本 - 均值) ** 2 / 方差)
            对数似然 = 对数归一化项 + 指数项
            对数概率列表.append(np.log(先验) + 对数似然)

        # 使用log-sum-exp技巧避免数值溢出
        最大对数概率 = np.max(对数概率列表)
        对数概率列表 = 对数概率列表 - 最大对数概率
        概率列表 = np.exp(对数概率列表)
        总概率 = np.sum(概率列表)
        return 概率列表 / (总概率 + 1e-10)  # 防止除以零


class Fisher线性判别器:
    """Fisher线性判别分析实现"""

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
    """K最近邻分类器实现"""

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


# ====================== 辅助功能函数 ======================
def 加载模型(模型路径):
    """加载序列化的模型对象"""
    try:
        with open(模型路径, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"❌ 加载模型失败: {str(e)}")
        return None


def 应用标准化(特征集, 均值, 标准差):
    """应用预计算的标准化参数"""
    return (特征集 - 均值) / (标准差 + 1e-8)


def 预测新数据(模型, 新数据, 标准化参数=None):
    """
    使用加载的模型进行预测
    :param 模型: 加载的模型对象
    :param 新数据: 待预测数据 (n_samples, n_features)
    :param 标准化参数: 包含'mean'和'std'的字典
    :return: 预测概率矩阵
    """
    # 应用标准化
    if 标准化参数:
        新数据 = 应用标准化(新数据, 标准化参数['mean'], 标准化参数['std'])

    # Fisher模型特殊处理
    if isinstance(模型, Fisher线性判别器):
        # 先投影再预测
        投影数据 = 模型.投影(新数据)
        # 需要加载fisher_knn_model进行预测
        knn模型 = 加载模型('fisher_knn_model.pkl')
        if knn模型 is None:
            raise ValueError("Fisher_KNN模型加载失败")
        return knn模型.预测概率(投影数据)

    # 其他模型直接预测
    if isinstance(模型, 贝叶斯决策器) or isinstance(模型, K近邻分类器):
        return np.array([模型.预测概率(样本) for 样本 in 新数据])

    raise TypeError(f"未知模型类型: {type(模型)}")


def 可视化预测结果(真实标签, 预测概率, 类别名称, 分类器名称):
    """
    可视化预测结果：混淆矩阵和分类报告
    """
    # 获取预测标签
    预测标签 = np.argmax(预测概率, axis=1)
    预测标签字符串 = [类别名称[i] for i in 预测标签]  # 按索引映射

    # 打印分类报告
    print(f"\n【{分类器名称}分类报告】")
    print(classification_report(真实标签, 预测标签字符串, target_names=类别名称, digits=4))

    # 绘制混淆矩阵
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
    # 1. 加载模型和参数
    print("=" * 50)
    print("开始加载模型和标准化参数...")
    开始时间 = time.time()

    贝叶斯模型 = 加载模型('bayes_model.pkl')
    fisher模型 = 加载模型('fisher_model.pkl')
    knn模型 = 加载模型('knn_model.pkl')

    # 加载标准化参数
    with open('scaler_params.pkl', 'rb') as f:
        scaler_params = pickle.load(f)

    加载时间 = time.time() - 开始时间
    print(f"✅ 模型加载完成，耗时: {加载时间:.2f}秒")
    print("=" * 50)

    # 2. 准备新数据
    print("请指定新数据文件路径（支持CSV/Excel格式）")
    数据路径 = input("请输入文件路径: ").strip() or '新数据路径.xlsx'

    try:
        if 数据路径.endswith('.csv'):
            新数据框 = pd.read_csv(数据路径)
        else:
            新数据框 = pd.read_excel(数据路径)

        # 提取特征（假设最后一列是标签）
        新特征 = 新数据框.iloc[:, :-1].values.astype(float)
        真实标签 = 新数据框.iloc[:, -1].values if 新数据框.shape[1] > 1 else None

        print(f"✅ 成功加载 {len(新特征)} 个样本，{新特征.shape[1]} 个特征")
    except Exception as e:
        print(f"❌ 数据加载失败: {str(e)}")
        exit(1)

    # 3. 进行预测
    print("\n" + "=" * 50)
    print("开始预测...")
    预测开始时间 = time.time()

    # 贝叶斯预测
    贝叶斯概率 = 预测新数据(贝叶斯模型, 新特征, scaler_params)

    # Fisher+KNN预测
    fisher概率 = 预测新数据(fisher模型, 新特征, scaler_params)

    # KNN预测
    knn概率 = 预测新数据(knn模型, 新特征, scaler_params)

    预测时间 = time.time() - 预测开始时间
    print(f"✅ 预测完成，总耗时: {预测时间:.2f}秒")
    print("=" * 50)


    # 4. 输出结果
    def 获取预测标签(概率矩阵):
        return np.argmax(概率矩阵, axis=1)


    print("\n预测结果摘要:")
    print(f"贝叶斯预测标签: {获取预测标签(贝叶斯概率)}")
    print(f"Fisher+KNN预测标签: {获取预测标签(fisher概率)}")
    print(f"K近邻预测标签: {获取预测标签(knn概率)}")

    # 5. 可视化结果（如果有真实标签）
    if 真实标签 is not None:
        类别名称 = [f"类别{i}" for i in range(贝叶斯概率.shape[1])]

        可视化预测结果(真实标签, 贝叶斯概率, 类别名称, "贝叶斯")
        可视化预测结果(真实标签, fisher概率, 类别名称, "Fisher+KNN")
        可视化预测结果(真实标签, knn概率, 类别名称, "K近邻")
    else:
        print("\n⚠️ 未提供真实标签，跳过评估可视化")

    # 6. 保存预测结果
    结果数据框 = pd.DataFrame({
        '贝叶斯预测': 获取预测标签(贝叶斯概率),
        'Fisher预测': 获取预测标签(fisher概率),
        'KNN预测': 获取预测标签(knn概率)
    })

    保存路径 = 数据路径.replace('.', '_预测结果.')
    结果数据框.to_csv(保存路径, index=False)
    print(f"\n💾 预测结果已保存至: {保存_path}")

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False