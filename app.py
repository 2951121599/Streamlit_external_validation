# 导入必要的库
import streamlit as st  # Streamlit用于创建Web应用界面
import pandas as pd     # 用于数据处理和分析
import numpy as np      # 用于数值计算
import tensorflow as tf  # 用于加载深度学习模型
import matplotlib.pyplot as plt  # 用于数据可视化
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc  # 用于计算评估指标
from matplotlib import rcParams  # 用于设置matplotlib的字体和样式
import io  # 用于处理内存中的文件对象
import os


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
    return tf.keras.models.load_model('data/MODEL.h5')

# 加载验证数据


@st.cache_data
def load_validation_data():
    """加载外部验证数据集"""
    df = pd.read_excel('data/merge_external_validation.xlsx')
    return df.iloc[:, :-2], df.iloc[:, -1]


# 模型的cutoff值
def find_model_cutoff():
    df = pd.read_csv('data/testset_cutoff.csv')
    y_true = df.iloc[:, 0]
    # 如果有多列，则计算每一列的cutoff值
    # # 创建cutoff数据框
    cutoff_df = pd.DataFrame()
    cutoff_df['model'] = df.columns[1:]
    y_pred = df.iloc[:, 1:]
    cutoff_values = []
    for col in y_pred.columns:
        fpr, tpr, thresholds = roc_curve(y_true, y_pred[col])
        cutoff = round(thresholds[np.argmax(tpr - fpr)], 3)
        cutoff_values.append(cutoff)
        cutoff_df[col] = cutoff
    return cutoff_df


# 计算评估指标的函数
def calculate_metrics(y_true, y_pred_proba, cutoff):
    # 根据模型的cutoff值计算ROC曲线
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    # 将概率转换为预测标签（阈值cutoff）
    y_pred = (y_pred_proba >= cutoff).astype(int)

    # 计算各项指标
    metrics = {
        'AUC': round(roc_auc, 3),  # AUC
        'Accuracy': round(accuracy_score(y_true, y_pred), 3),    # 准确率
        'Precision': round(precision_score(y_true, y_pred), 3),  # 精确率
        'Recall': round(recall_score(y_true, y_pred), 3),        # 召回率
        'F1 Score': round(f1_score(y_true, y_pred), 3),           # F1分数
        'Cutoff': round(cutoff, 3)  # cutoff
    }

    return metrics, fpr, tpr, roc_auc, y_pred, cutoff


# 主程序
def main():
    # 加载模型和数据
    model = load_model()
    X_val, y_true = load_validation_data()
    print(f"X_val: {X_val}")
    print(f"y_true: {y_true}")

    # 主要内容
    st.header("Model Performance Metrics")  # 模型性能指标

    # 计算cutoff值
    cutoff_df = find_model_cutoff()
    
    # 显示cutoff数据框
    st.write("Model Cutoff:")
    st.dataframe(cutoff_df)
        
    model_cutoff = cutoff_df.iloc[0, 1]
    print(f"model_cutoff: {model_cutoff}")

    # 进行预测
    y_pred_proba = model.predict(X_val, verbose=0)

    # 如果有真实标签，计算评估指标
    if y_true is not None:
        # 计算评估指标
        metrics, fpr, tpr, roc_auc, y_pred, cutoff = calculate_metrics(
            y_true, y_pred_proba, model_cutoff)

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

        # 保存ROC曲线为PDF
        buf = io.BytesIO()
        fig.savefig(buf, format='pdf', bbox_inches='tight', dpi=300)
        buf.seek(0)

        # 显示ROC曲线
        st.pyplot(fig)

        # 添加PDF下载按钮
        st.download_button(
            label="Download ROC Curve (PDF)",
            data=buf,
            file_name='roc_curve.pdf',
            mime='application/pdf'
        )

        # 关闭图形，释放内存
        plt.close(fig)

        # 创建评估结果数据框
        st.header("Evaluation Results")

        # 创建指标数据框
        metrics_df = pd.DataFrame({
            'Metric': ['AUC', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'Cutoff'],
            'Value': [metrics['AUC'], metrics['Accuracy'], metrics['Precision'],
                      metrics['Recall'], metrics['F1 Score'], metrics['Cutoff']]
        })

        # 设置显示格式为3位小数
        pd.set_option('display.float_format', lambda x: '%.3f' % x)

        # 显示指标数据框
        st.write("Model Performance Metrics:")
        st.dataframe(metrics_df)

        # 创建预测结果数据框
        results_df = pd.DataFrame({
            'True Label': y_true,  # 真实标签
            'Predicted Label': y_pred.flatten(),  # 预测标签
            'Predicted Probability': y_pred_proba.flatten()  # 预测概率
        })

        # 显示预测结果数据框
        st.write("Prediction Results:")
        st.dataframe(results_df)

        # 添加下载按钮
        st.download_button(
            label="Download Evaluation Results",
            data=metrics_df.to_csv(index=False).encode('utf-8'),
            file_name='evaluation_metrics.csv',
            mime='text/csv',
        )

        st.download_button(
            label="Download Prediction Results",
            data=results_df.to_csv(index=False).encode('utf-8'),
            file_name='prediction_results.csv',
            mime='text/csv',
        )

    # 页脚
    st.markdown("---")
    st.markdown(
        f"External Validation Analysis | Updated: {pd.Timestamp.now().strftime('%Y-%m-%d')}")


if __name__ == "__main__":
    main()
