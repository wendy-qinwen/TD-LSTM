import numpy as np 


import numpy as np
from scipy import stats
from sklearn.metrics import r2_score




class Metrics:
    def __init__(self):
        pass

    def mae(self, y_true, y_pred):
        """
        Calculate Mean Absolute Error (MAE)
        """
        return np.mean(np.abs(y_true - y_pred))

    def rmse(self, y_true, y_pred):
        """
        Calculate Root Mean Square Error (RMSE)
        """
        return np.sqrt(np.mean((y_true - y_pred) ** 2))

    def wmape(self, y_true, y_pred):
        """
        Calculate Weighted Mean Absolute Percentage Error (WMAPE)
        """
        return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))

    def r_squared(self, y_true, y_pred):
        """
        Calculate R-squared (Coefficient of Determination)
        """
        # ss_res = np.sum((y_true - y_pred) ** 2)
        # ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

        # 假设 y_true 是真实数据，y_pred 是预测数据
        # 计算 R²
        r2 = r2_score(y_true, y_pred)

        # 计算显著性（p值）
        # 使用 t 检验
        t_stat, p_value = stats.ttest_ind(y_true, y_pred)

        # 或者使用相关系数检验
        corr, p_value = stats.pearsonr(y_true, y_pred)

        # 打印结果
        # print(f'R² = {r2:.4f}')
        # print(f'p-value = {p_value:.4f}')

        # 判断显著性
        alpha = 0.05  # 显著性水平
        # if p_value < alpha:
        #     print('结果显著')
        # else:
        #     print('结果不显著')
        return r2,p_value


    def MetricsAll(self,y_true,y_pred):
        """
        
        """
        return (self.mae(y_true,y_pred), self.rmse(y_true,y_pred),self.wmape(y_true,y_pred),self.r_squared(y_true,y_pred)[0],self.r_squared(y_true,y_pred)[1])

