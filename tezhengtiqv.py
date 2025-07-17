import pandas as pd
import numpy as np
from sklearn.decomposition import PCA#主成分分析，用于降维
from sklearn.feature_selection import SelectKBest, RFE, SelectFromModel#基于单变量统计检验选择最佳特征，递归特征消除，基于模型的特征选择。
from sklearn.feature_selection import f_classif, mutual_info_classif#特征选择的统计方法
from sklearn.ensemble import RandomForestClassifier#随机森林分类器
from sklearn.preprocessing import StandardScaler#数据标准化
import os#处理文件路径


# ====================== 配置文件 ======================
class 特征工程配置:
    """配置特征工程参数"""

    def __init__(self):
        # 特征选择方法 (可选: 'kbest'过滤式, 'rfe'包裹式, 'embedded'嵌入式)
        self.特征选择方法 = 'embedded'

        # 特征提取方法 (可选: 'pca'主成分分析, None)
        self.特征提取方法 = 'pca'

        # 输出文件名
        self.输出文件路径 = 'processed_features.csv'

        # 特征选择参数
        self.选择特征数 = 10  # 选择TopN特征
        self.随机种子 = 42  # 确保结果可复现


# ====================== 核心功能 ======================
def 执行特征工程(原始特征, 标签, 配置):
    """
    执行特征工程流程
    :param 原始特征: 输入特征矩阵 (n_samples, n_features)
    :param 标签: 目标标签向量 (n_samples,)
    :param 配置: 特征工程配置对象
    :return: 处理后的特征矩阵
    """
    # 1. 数据标准化
    标准化器 = StandardScaler()
    标准特征 = 标准化器.fit_transform(原始特征)

    # 2. 特征选择
    if 配置.特征选择方法 == 'kbest':
        print(f"✅ 使用过滤式特征选择 (SelectKBest), 选择Top{配置.选择特征数}特征")
        选择器 = SelectKBest(score_func=f_classif, k=配置.选择特征数)
        选择后特征 = 选择器.fit_transform(标准特征, 标签)
        特征掩码 = 选择器.get_support()

    elif 配置.特征选择方法 == 'rfe':
        print(f"✅ 使用包裹式特征选择 (RFE), 选择Top{配置.选择特征数}特征")
        基模型 = RandomForestClassifier(n_estimators=100, random_state=配置.随机种子)
        选择器 = RFE(estimator=基模型, n_features_to_select=配置.选择特征数)
        选择后特征 = 选择器.fit_transform(标准特征, 标签)
        特征掩码 = 选择器.get_support()

    elif 配置.特征选择方法 == 'embedded':
        print(f"✅ 使用嵌入式特征选择 (随机森林重要性), 选择Top{配置.选择特征数}特征")
        基模型 = RandomForestClassifier(n_estimators=100, random_state=配置.随机种子)
        基模型.fit(标准特征, 标签)
        选择器 = SelectFromModel(基模型, max_features=配置.选择特征数, threshold=-np.inf)
        选择后特征 = 选择器.fit_transform(标准特征, 标签)
        特征掩码 = 选择器.get_support()
    else:
        print("⚠️ 未启用特征选择，使用全部原始特征")
        选择后特征 = 标准特征
        特征掩码 = np.ones(标准特征.shape[1], dtype=bool)

    # 3. 特征提取
    if 配置.特征提取方法 == 'pca':
        print(f"✅ 应用PCA特征提取, 降维至{配置.选择特征数}主成分")
        pca = PCA(n_components=配置.选择特征数, random_state=配置.随机种子)
        最终特征 = pca.fit_transform(选择后特征)
        print(f"  主成分解释方差比: {np.sum(pca.explained_variance_ratio_):.2%}")
    else:
        最终特征 = 选择后特征

    return 最终特征, 特征掩码


# ====================== 文件处理 ======================
def 保存处理结果(处理特征, 标签, 特征名称, 文件路径):
    """
    保存处理后的特征到CSV文件
    :param 处理特征: 处理后的特征矩阵
    :param 标签: 目标标签向量
    :param 特征名称: 原始特征名称列表
    :param 文件路径: 输出文件路径
    """
    # 生成新特征名称
    if len(处理特征.shape) == 1:
        新特征名 = ['Feature_1']
    else:
        新特征名 = [f'Feature_{i + 1}' for i in range(处理特征.shape[1])]

    # 创建DataFrame
    结果数据 = pd.DataFrame(处理特征, columns=新特征名)
    结果数据['Target'] = 标签

    # 保存文件
    结果数据.to_csv(文件路径, index=False)
    print(f"  已保存处理结果至: {os.path.abspath(文件路径)}")
    print(f"  新数据集维度: {结果数据.shape}")


# ====================== 主程序 ======================
if __name__ == "__main__":
    # ========== 配置区域 ==========
    配置 = 特征工程配置()#使用Embedded，pca，输出特征为10
    # ========== 数据加载 ==========
    文件路径 = r'D:\应用数据\onedive\OneDrive\Desktop\1\DryBeanDataset\DryBeanDataset\Dry_Bean_Dataset.xlsx'

    try:
        print(f" 加载数据: {文件路径}")
        数据框 = pd.read_excel(文件路径)

        # 分离特征和标签
        原始特征 = 数据框.iloc[:, :-1].values.astype(float)
        标签 = 数据框.iloc[:, -1].values
        特征名称 = 数据框.columns[:-1].tolist()

        print(f"  原始数据集维度: {原始特征.shape}")
        print(f"  特征数量: {len(特征名称)}")
        print(f"  样本数量: {len(标签)}")

        # ========== 执行特征工程 ==========
        处理后特征, 特征掩码 = 执行特征工程(原始特征, 标签, 配置)

        # ========== 保存结果 ==========
        保存处理结果(处理后特征, 标签, 特征名称, 配置.输出文件路径)

        # ========== 输出特征报告 ==========
        print("\n🔍 特征工程报告:")#hasattr()是一个内置函数，用于检查对象是否具有指定的属性或方法
        if hasattr(配置, '特征选择方法') and 配置.特征选择方法 != 'none':
            选中特征数 = np.sum(特征掩码)
            print(f"  特征选择: 从 {len(特征掩码)} 个特征中选中 {选中特征数} 个关键特征")

        if hasattr(配置, '特征提取方法') and 配置.特征提取方法:
            print(f"  特征提取: 使用{配置.特征提取方法.upper()}生成新特征空间")

        print(f"  最终特征维度: {处理后特征.shape[1]}")

    except Exception as e:
        print(f"❌ 数据处理错误: {str(e)}")
        import traceback

        traceback.print_exc()