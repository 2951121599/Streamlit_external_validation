# 导入必要的库
import streamlit as st  # Streamlit用于创建Web应用界面
import pandas as pd     # 用于数据处理和分析
import numpy as np      # 用于数值计算
import tensorflow as tf  # 用于加载深度学习模型
import matplotlib.pyplot as plt  # 用于数据可视化
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc  # 用于计算评估指标
from matplotlib import rcParams  # 用于设置matplotlib的字体和样式

# 配置matplotlib的显示参数
rcParams['font.family'] = 'sans-serif'  # 设置字体
rcParams['font.size'] = 12              # 设置字体大小

# 设置Streamlit页面配置
st.set_page_config(
    page_title="External Validation Analysis",  # 页面标题
    page_icon="📊",                            # 页面图标
    layout="wide",                             # 使用宽屏布局
    initial_sidebar_state="expanded"           # 初始侧边栏状态为展开
)

# 自定义CSS样式
st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}  # 主背景色
    .stButton>button {background-color: #007bff; color: white; border-radius: 8px;}  # 按钮样式
    .metric-card {background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);}  # 指标卡片样式
    h1 {color: #2c3e50; text-align: center;}  # 标题样式
    h2 {color: #34495e; border-bottom: 2px solid #17a2b8; padding-bottom: 5px;}  # 二级标题样式
    </style>
""", unsafe_allow_html=True)

# 页面标题和介绍
st.title("📊 External Validation Analysis")
st.markdown("""
    此工具使用独立数据集对预测模型进行外部验证。
    计算关键评估指标并生成ROC曲线以评估模型性能。
""")

# 加载预训练模型


@st.cache_resource
def load_model():
    """加载预训练的深度学习模型"""
    return tf.keras.models.load_model('data/MODEL_2025_05_16_19_37_41.h5')

# 加载验证数据


@st.cache_data
def load_validation_data():
    """加载外部验证数据集"""
    df = pd.read_excel('data/merge_external_validation.xlsx')
    return df.iloc[:, :-2], df.iloc[:, -1]

# 计算评估指标的函数


def calculate_metrics(y_true, y_pred_proba, cutoff):
    """
    计算模型性能指标

    参数:
    y_true: 真实标签
    y_pred_proba: 预测概率

    返回:
    metrics: 包含各项指标的字典
    fpr, tpr: ROC曲线的假阳性率和真阳性率
    roc_auc: ROC曲线下面积
    """
    # 将概率转换为预测标签（阈值cutoff）
    y_pred = (y_pred_proba >= cutoff).astype(int)

    # 根据模型的cutoff值计算ROC曲线
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    # 计算各项指标
    metrics = {
        'AUC': roc_auc,  # AUC
        'Accuracy': accuracy_score(y_true, y_pred),    # 准确率
        'Precision': precision_score(y_true, y_pred),  # 精确率
        'Recall': recall_score(y_true, y_pred),        # 召回率
        'F1 Score': f1_score(y_true, y_pred),           # F1分数
        'Cutoff': cutoff  # cutoff
    }

    return metrics, fpr, tpr, roc_auc, y_pred


# 主程序
def main():
    # 加载模型和数据
    model = load_model()
    X_val, y_true = load_validation_data()
    cutoff = 0.587

    # 主要内容
    st.header("Model Performance Metrics")  # 模型性能指标

    # 进行预测
    y_pred_proba = model.predict(X_val, verbose=0)

    # 如果有真实标签，计算评估指标
    if y_true is not None:
        # 计算评估指标
        metrics, fpr, tpr, roc_auc, y_pred = calculate_metrics(y_true, y_pred_proba, cutoff)

        # 在网格中显示指标，增加auc,acc,precision,recall,f1,cutoff
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        with col1:
            st.metric("AUC", f"{metrics['AUC']:.3f}")    # 显示AUC
        with col2:
            st.metric("Accuracy", f"{metrics['Accuracy']:.3f}")    # 显示准确率
        with col3:
            st.metric("Precision", f"{metrics['Precision']:.3f}")  # 显示精确率
        with col4:
            st.metric("Recall", f"{metrics['Recall']:.3f}")        # 显示召回率
        with col5:
            st.metric("F1 Score", f"{metrics['F1 Score']:.3f}")    # 显示F1分数
        with col6:
            st.metric("Cutoff", f"{cutoff:.3f}")    # 显示cutoff
        # 绘制ROC曲线
        st.header("ROC Curve")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.3f})')  # 绘制ROC曲线
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # 绘制对角线
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')  # 设置x轴标签
        ax.set_ylabel('True Positive Rate')   # 设置y轴标签
        ax.set_title('External Validation ROC Curve')  # 设置标题
        ax.legend(loc="lower right")  # 添加图例
        st.pyplot(fig)

    # 显示详细结果
    st.header("Detailed Results")
    # 创建结果数据框
    results_df = pd.DataFrame({
        'True Label': y_true,  # 真实标签
        'Predicted Label': y_pred.flatten(),  # 预测标签
        'Predicted Probability': y_pred_proba.flatten()  # 预测概率
    })

    st.dataframe(results_df)  # 显示结果表格

    # 页脚
    st.markdown("---")
    st.markdown(
        f"External Validation Analysis | Updated: {pd.Timestamp.now().strftime('%Y-%m-%d')}")


if __name__ == "__main__":
    main()
