import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cấu hình trang
st.set_page_config(page_title="Phân Tích Bệnh Phổi", page_icon="🫁", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv('lung_disease_data_cleaned.csv', sep=',')
    return df

# Bảng màu thân thiện với người mù màu
COLOR_PALETTE = [
    '#1F77B4',  # Xanh dương
    '#FF7F0E',  # Cam
    '#2CA02C',  # Xanh lá
    '#D62728',  # Đỏ
    '#9467BD',  # Tím
    '#8C564B',  # Nâu
    '#E377C2',  # Hồng
    '#7F7F7F',  # Xám
]

def plot_recovery_by_disease(df, chart_type='pie'):
    # Tính tỷ lệ phần trăm phục hồi theo loại bệnh
    recovery_rates = df.groupby('Loại Bệnh')['Phục Hồi'].value_counts(normalize=True).unstack() * 100
    
    if chart_type == 'stacked':
        # Stacked bar chart
        plt.figure(figsize=(10, 6))
        recovery_rates.plot(kind='bar', stacked=True, color=['#FF6B6B', '#4ECDC4'])
        plt.title('Tỷ lệ phục hồi theo loại bệnh', fontsize=14, pad=15)
        plt.xlabel('Loại bệnh', fontsize=12)
        plt.ylabel('Tỷ lệ (%)', fontsize=12)
        plt.legend(title='Phục Hồi', loc='center left', bbox_to_anchor=(1.0, 0.5))
        plt.xticks(rotation=45, ha='right')
        plt.yticks(ticks=range(0, 101, 10), labels=[f"{i}%" for i in range(0, 101, 10)])
        plt.tight_layout()
    else:  # default to pie chart
        # Pie chart cho từng loại bệnh
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))  #
        axes = axes.flatten()  
        diseases = recovery_rates.index
        for idx, disease in enumerate(diseases):
            if idx < len(axes):
                axes[idx].pie(recovery_rates.loc[disease], labels=recovery_rates.columns,
                             autopct='%1.1f%%', colors=['#FF6B6B', '#4ECDC4'], startangle=90)
                axes[idx].set_title(f'{disease}', fontsize=10)
        for idx in range(len(diseases), len(axes)):
            axes[idx].axis('off')
        plt.suptitle('Tỷ lệ phục hồi theo loại bệnh', fontsize=14, y=1.05)
        plt.tight_layout()
    
    return plt

# Hàm tạo biểu đồ ảnh hưởng của hút thuốc (stacked bar chart hoặc pie chart)
def plot_smoking_impact(df, chart_type='stacked'):
    # Tính tỷ lệ phần trăm phục hồi theo tình trạng hút thuốc
    smoking_impact = df.groupby('Tình Trạng Hút Thuốc')['Phục Hồi'].value_counts(normalize=True).unstack() * 100
    
    if chart_type == 'stacked':
        # Stacked bar chart
        plt.figure(figsize=(8, 6))
        smoking_impact.plot(kind='bar', stacked=True, color=['#FF6B6B', '#4ECDC4'])
        plt.title('Ảnh hưởng của hút thuốc đến khả năng phục hồi', fontsize=14, pad=15)
        plt.xlabel('Tình trạng hút thuốc', fontsize=12)
        plt.ylabel('Tỷ lệ (%)', fontsize=12)
        plt.legend(title='Phục Hồi', loc='center left', bbox_to_anchor=(1.0, 0.5))
        plt.xticks(rotation=0)
        # Chia trục y thành 10 khoảng từ 0% đến 100%
        plt.yticks(ticks=range(0, 101, 10), labels=[f"{i}%" for i in range(0, 101, 10)])
        plt.tight_layout()
        return plt
    else:  # Pie chart
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        for idx, smoking_status in enumerate(smoking_impact.index):
            axes[idx].pie(smoking_impact.loc[smoking_status], labels=smoking_impact.columns,
                          autopct='%1.1f%%', colors=['#FF6B6B', '#4ECDC4'], startangle=90)
            axes[idx].set_title(f'Tình trạng: {smoking_status}', fontsize=12)
        plt.suptitle('Ảnh hưởng của hút thuốc đến khả năng phục hồi', fontsize=14, y=1.05)
        plt.tight_layout()
        return plt

# Hàm chính
def main():
    st.title('Phân tích dữ liệu bệnh phổi')
    
    # Đọc dữ liệu
    df = load_data()
    if df is None:
        return
    
    # Sidebar để tùy chỉnh
    st.sidebar.header('Tùy chỉnh biểu đồ')
    chart_options_1 = ['stacked', 'pie']
    chart_options_2 = ['stacked', 'pie']
    
    # Tùy chọn cho biểu đồ 
    chart_type_1 = st.sidebar.selectbox('Chọn loại biểu đồ cho "Tỷ lệ phục hồi theo loại bệnh:', 
                                       chart_options_2, index=0)
    chart_type_2 = st.sidebar.selectbox('Chọn loại biểu đồ cho "Ảnh hưởng của hút thuốc":', 
                                       chart_options_2, index=0)
    
    # Hiển thị dữ liệu cơ bản
    st.subheader('Dữ liệu mẫu')
    st.write(df.head())
    
    # Câu hỏi 1
    st.subheader('Câu hỏi 1: Tỷ lệ phục hồi theo loại bệnh')
    st.write("""
    Phân tích này giúp chúng ta hiểu loại bệnh nào có khả năng phục hồi cao hơn,
    từ đó đưa ra chiến lược điều trị phù hợp.
    """)
    fig1 = plot_recovery_by_disease(df, chart_type_1)
    if fig1:
        st.pyplot(fig1)
    
    # Thêm khoảng cách giữa các biểu đồ
    st.markdown("---")
    
    # Câu hỏi 2
    st.subheader('Câu hỏi 2: Ảnh hưởng của hút thuốc đến khả năng phục hồi')
    st.write("""
    Phân tích này xem xét mối quan hệ giữa tình trạng hút thuốc và khả năng phục hồi.
    """)
    fig2 = plot_smoking_impact(df, chart_type_2)
    if fig2:
        st.pyplot(fig2)
    
    # Thống kê cơ bản
    st.subheader('Thống kê cơ bản')
    st.write(df.describe())

if __name__ == '__main__':
    main()