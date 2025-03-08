import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sklearn.preprocessing as skp
import plotly.express as px

# Thiết lập cấu hình trang
st.set_page_config(page_title="Phân Tích Bệnh Phổi", page_icon="🫁", layout="wide", initial_sidebar_state="expanded")

# Thiết lập bảng màu
sns.set_palette("colorblind")
PALETTE = ["#88CCEE", "#44AA99", "#117733", "#999933", "#DDCC77", "#CC6677", "#882255", "#AA4499"]

# --- Hàm tải và tiền xử lý dữ liệu ---
@st.cache_data(show_spinner=False)
def load_and_preprocess_data(file):
    try:
        # Đọc dữ liệu từ tệp được tải lên
        df = pd.read_csv(file, na_values=["None", " ", "UNKNOWN", -1, 999, "NA", "N/A", "NULL", ""])
        
        # Xác định tên cột có thể bằng tiếng Anh hoặc tiếng Việt
        col_mapping = {
            "Age": "Tuổi", "Gender": "Giới Tính", "Smoking Status": "Tình Trạng Hút Thuốc",
            "Lung Capacity": "Dung Tích Phổi", "Disease Type": "Loại Bệnh", 
            "Treatment Type": "Loại Điều Trị", "Hospital Visits": "Số Lượt Khám Bệnh", 
            "Recovered": "Hồi Phục"
        }
        df.columns = [col_mapping.get(col, col) for col in df.columns]
        
        # Chuyển đổi các cột sang kiểu số
        numeric_cols = ["Tuổi", "Dung Tích Phổi", "Số Lượt Khám Bệnh"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Chuyển đổi cột Hồi Phục sang kiểu số (0/1)
        df["Hồi Phục"] = df["Hồi Phục"].map({"Có": 1, "Yes": 1, "Không": 0, "No": 0})
        
        # Đảm bảo các cột phân loại được xử lý đúng
        categorical_cols = ["Giới Tính", "Tình Trạng Hút Thuốc", "Loại Bệnh", "Loại Điều Trị"]
        for col in categorical_cols:
            df[col] = df[col].astype('category')
        return df
    except Exception as e:
        st.error(f"Lỗi khi xử lý dữ liệu: {e}")
        st.stop()

# Hàm mã hóa dữ liệu phân loại (chỉ dùng cho tương quan)
def encode_data(data):
    encoder = skp.LabelEncoder()
    encoded_data = data.copy()
    for col in data.select_dtypes(include=['object', 'category']).columns:
        encoded_data[col] = encoder.fit_transform(data[col].astype(str))
    return encoded_data

# Hàm vẽ đồ thị tỷ lệ phục hồi bệnh
def plot_recovery_by_disease(df, chart_type='Pie'):
    # Tính tỷ lệ phần trăm phục hồi theo loại bệnh
    recovery_rates = df.groupby('Loại Bệnh')['Hồi Phục'].value_counts(normalize=True).unstack() * 100
    
    if chart_type == 'Stacked':
        # Stacked bar chart
        plt.figure(figsize=(10, 6))
        recovery_rates.plot(kind='bar', stacked=True, color=['#FF6B6B', '#4ECDC4'])
        plt.title('Tỷ lệ phục hồi theo loại bệnh', fontsize=14, pad=15)
        plt.xlabel('Loại bệnh', fontsize=12)
        plt.ylabel('Tỷ lệ (%)', fontsize=12)
        plt.legend(title='Hồi phục', loc='center left', bbox_to_anchor=(1.0, 0.5))
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
def plot_smoking_impact(df, chart_type='Stacked'):
    # Tính tỷ lệ phần trăm phục hồi theo tình trạng hút thuốc
    smoking_impact = df.groupby('Tình Trạng Hút Thuốc')['Hồi Phục'].value_counts(normalize=True).unstack() * 100
    
    if chart_type == 'Stacked':
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
    
    
# --- Tải dữ liệu từ người dùng ---
st.sidebar.header("Tải Dữ Liệu")
uploaded_file = st.sidebar.file_uploader("Tải lên tệp CSV", type=["csv"])
if uploaded_file is not None:
    df = load_and_preprocess_data(uploaded_file)
else:
    st.warning("Vui lòng tải lên tệp CSV để tiếp tục.")
    st.stop()

# --- CSS tùy chỉnh cho giao diện ---
st.markdown("""
    <style>
    div[role="radiogroup"] label {
        display: flex;
        align-items: center;
        font-size: 18px;
        font-weight: bold;
        color: #333;
        padding: 10px 15px;
        border-radius: 8px;
        transition: background-color 0.3s ease, transform 0.2s ease;
        cursor: pointer;
    }
    div[role="radiogroup"] label span {
        visibility: hidden;
        width: 0;
        margin: 0;
        padding: 0;
    }
    div[role="radiogroup"] label:hover {
        background-color: #FF851B;
        color: white !important;
        transform: scale(1.02);
    }
    div[role="radiogroup"] label[data-selected="true"] {
        background-color: #FF851B !important;
        color: white !important;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# --- Tiêu đề và điều hướng sidebar ---
st.title("🫁 Bảng Điều Khiển Phân Tích Bệnh Phổi")
page = st.sidebar.radio("Chọn Trang", [
    "1. Giới Thiệu Dữ Liệu", 
    "2. Thống Kê Mô Tả", 
    "3. Phân Tích Chuyên Sâu", 
    "4. Nhận Xét Chung", 
], index=0)

# --- Trang 1: Giới Thiệu Dữ Liệu ---
if page == "1. Giới Thiệu Dữ Liệu":
    st.header("1. Giới Thiệu Dữ Liệu")
    st.subheader("Nguồn Gốc Dữ Liệu")
    st.markdown("""
    - Dữ liệu được lấy từ nền tảng Kaggle: [Lung Disease Prediction](https://www.kaggle.com/datasets/samikshadalvi/lungs-diseases-dataset).
    - Tập dữ liệu bao gồm thông tin về bệnh nhân mắc các bệnh phổi như hen suyễn, viêm phế quản, COPD, ung thư phổi, và viêm phổi.
    - Dữ liệu bao gồm các thông tin nhân khẩu học, tình trạng hút thuốc, dung tích phổi, số lượt khám bệnh, và kết quả hồi phục.
    """)

    st.subheader("Mô Tả Dữ Liệu")
    st.markdown("""
    - **🧑‍🤝‍🧑Age:** Tuổi của bệnh nhân (số nguyên).
    - **♀️Gender:** Giới tính (Male/Female).
    - **🚬Smoking Status:** Tình trạng hút thuốc (Yes/No).
    - **🌡️Lung Capacity:** Dung tích phổi của bệnh nhân (số thực, đơn vị lít).
    - **🫁Disease Type:** Loại bệnh phổi (Asthma, Bronchitis, COPD, Lung Cancer, Pneumonia).
    - **💊Treatment Type:** Loại điều trị (Medication, Surgery, Therapy).
    - **🏥Hospital Visits:** Số lượt khám bệnh (số nguyên).
    - **✅Recovered:** Bệnh nhân đã hồi phục chưa? (0: No, 1: Yes).
    """)

    with st.expander("Xem Toàn Bộ Dữ Liệu Thô (đã dịch sang tiếng Việt)"):
        st.dataframe(df)

# --- Trang 2: Thống Kê Mô Tả ---
elif page == "2. Thống Kê Mô Tả":
    st.header("2. Thống Kê Mô Tả")
    st.subheader("Thông Tin Dữ Liệu")
    st.dataframe(df.describe())

    st.subheader("Phân Phối Dữ Liệu")
    numeric_cols = df.select_dtypes(include=np.number).columns
    cat_cols = df.select_dtypes(include=['category']).columns.tolist()

    # Phân phối biến số
    num_select = st.selectbox("Chọn một biến số để xem phân phối biến của biến numerical:", numeric_cols)
    if num_select:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df[num_select].dropna(), kde=True, color=PALETTE[0], ax=ax)
        ax.set_title(f"Phân bố của {num_select}")
        st.pyplot(fig)

    with st.expander("Nhận xét về Thống Kê Mô Tả"):
        st.markdown("""
        - **Tuổi:** Tuổi trung bình, độ lệch chuẩn, và phạm vi tuổi của bệnh nhân.
        - **Dung Tích Phổi:** Dung tích phổi trung bình và phân phối (đơn vị lít).
        - **Số Lượt Khám Bệnh:** Số lượt khám trung bình.
        - **Hồi Phục:** Tỷ lệ hồi phục (0: Không, 1: Có).
        """)

    # Phân phối biến phân loại
    cat_select = st.selectbox("Chọn một biến phân loại để xem phân phối của biến categorical:", cat_cols)
    if cat_select:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(x=df[cat_select], palette=PALETTE, ax=ax)
        ax.set_title(f"Phân bố của {cat_select}")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig)


# --- Trang 3: Phân Tích Chuyên Sâu ---
elif page == "3. Phân Tích Chuyên Sâu":
    st.header("3. Phân Tích Chuyên Sâu")
    analysis_page = st.selectbox("Chọn Phân Tích", [
        "Thống kê chung", 
        "Tuổi & Dung Tích Phổi", 
        "Dung Lượng Phổi Trung Bình Theo Nhóm Tuổi và Loại Bệnh",
        "Loại Bệnh", 
        "Hút Thuốc & Dung Tích Phổi", 
        "Lượt Khám Bệnh", 
        "Tương Quan",
        "Phân Tích Song Biến (Bivariate Analysis)",
        "Tỷ lệ hồi phục"
    ])
    
    # Thống kê chung
    if analysis_page == "Thống kê chung":
        st.subheader("Thống Kê Chung")
        
        # Không loại bỏ NaN ở đây để giữ thông tin tổng quát, bao gồm cả giá trị thiếu
        total_patients = len(df)
        avg_age = df["Tuổi"].mean()  # mean() tự động xử lý NaN
        avg_lung_capacity = df["Dung Tích Phổi"].mean()
        smoking_rate = (df["Tình Trạng Hút Thuốc"] == "Có").mean() * 100

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Tổng Số Bệnh Nhân", total_patients)
        with col2:
            st.metric("Tuổi Trung Bình", f"{avg_age:.2f}" if not pd.isna(avg_age) else "N/A")
        with col3:
            st.metric("Dung Tích Phổi Trung Bình", f"{avg_lung_capacity:.2f}" if not pd.isna(avg_lung_capacity) else "N/A")
        with col4:
            st.metric("Tỷ Lệ Hút Thuốc (%)", f"{smoking_rate:.2f}" if not pd.isna(smoking_rate) else "N/A")

        with st.expander("Xem Nhận Xét Chi Tiết"):
            st.markdown(f"""
            - **Tổng Số Bệnh Nhân:** {total_patients}
                - Cho biết quy mô của tập dữ liệu.
            - **Tuổi Trung Bình:** {f'{avg_age:.2f}' if not pd.isna(avg_age) else 'N/A'}
                - Độ tuổi trung bình của bệnh nhân, phản ánh đặc điểm dân số nghiên cứu.
            - **Dung Tích Phổi Trung Bình:** {f'{avg_lung_capacity:.2f}' if not pd.isna(avg_lung_capacity) else 'N/A'}
                - Dung tích phổi trung bình, có thể bị ảnh hưởng bởi bệnh lý hoặc hút thuốc.
            - **Tỷ Lệ Hút Thuốc:** {f'{smoking_rate:.2f}%' if not pd.isna(smoking_rate) else 'N/A'}
                - Phần trăm bệnh nhân hút thuốc, một yếu tố nguy cơ quan trọng đối với bệnh phổi.
            """)
    
    # Phân bố Tuổi & Dung Tích Phổi
    elif analysis_page == "Tuổi & Dung Tích Phổi":
        st.subheader("Phân Bố Tuổi & Dung Tích Phổi")

        # Xử lý giá trị thiếu trước khi lọc
        filtered_df = df.dropna(subset=["Tuổi", "Dung Tích Phổi"])

        # Thanh trượt chọn khoảng tuổi (nếu có dữ liệu hợp lệ)
        if not filtered_df["Tuổi"].isnull().all():
            age_range = st.slider("Chọn Khoảng Tuổi", int(filtered_df["Tuổi"].min()), int(filtered_df["Tuổi"].max()), 
                                  (int(filtered_df["Tuổi"].min()), int(filtered_df["Tuổi"].max())))
            filtered_df = filtered_df[(filtered_df["Tuổi"] >= age_range[0]) & (filtered_df["Tuổi"] <= age_range[1])]
        else:
            st.warning("Không có dữ liệu tuổi hợp lệ để hiển thị.")
            st.stop()

        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        sns.histplot(filtered_df["Tuổi"], bins=20, kde=True, ax=ax[0], color=PALETTE[0])
        ax[0].set_title("Phân Bố Tuổi")
        ax[0].set_xlabel("Tuổi")
        ax[0].set_ylabel("Số Lượng")
        sns.histplot(filtered_df["Dung Tích Phổi"], bins=20, kde=True, ax=ax[1], color=PALETTE[1])
        ax[1].set_title("Phân Bố Dung Tích Phổi")
        ax[1].set_xlabel("Dung Tích Phổi (lít)")
        ax[1].set_ylabel("Số Lượng")
        st.pyplot(fig)

        with st.expander("Xem Nhận Xét Chi Tiết"):
            st.markdown(f"""
            **Biểu đồ Phân Bố Tuổi:**
            - Hiển thị sự phân bố độ tuổi của bệnh nhân (sau khi lọc và loại bỏ giá trị thiếu).
            - **Hình dạng phân bố:** Quan sát phân bố đối xứng, lệch trái, lệch phải hay đa đỉnh. Ví dụ, phân bố lệch phải cho thấy nhiều bệnh nhân lớn tuổi hơn.
            - **Độ tập trung:** Xác định khoảng tuổi phổ biến nhất (đỉnh của biểu đồ).
            - **Đường KDE:** Đường cong KDE ước lượng mật độ xác suất, cho thấy xu hướng chung.

            **Biểu đồ Phân Bố Dung Tích Phổi:**
            - Hiển thị sự phân bố dung tích phổi (lít) của bệnh nhân.
            - **Hình dạng phân bố:** Quan sát tính đối xứng hoặc lệch của phân bố.
            - **Độ tập trung:** Xác định khoảng dung tích phổi phổ biến nhất.
            - **Độ phân tán:** Phân bố rộng cho thấy sự khác biệt lớn về dung tích phổi; phân bố hẹp cho thấy sự đồng đều.
            - **Đường KDE:** Đường cong KDE giúp hình dung xu hướng chung.
            - **So sánh với tuổi:** Có thể có mối liên hệ giữa tuổi và dung tích phổi (ví dụ: dung tích giảm khi tuổi tăng).
            """)

    # Phân bố Loại Bệnh
    elif analysis_page == "Loại Bệnh":
        st.subheader("Phân Bố Loại Bệnh")
        disease_counts = df["Loại Bệnh"].value_counts()
        st.bar_chart(disease_counts)

        with st.expander("Xem Nhận Xét Chi Tiết"):
            st.markdown("""
            **Biểu đồ Cột Phân Bố Loại Bệnh:**
            - Thể hiện số lượng bệnh nhân mắc mỗi loại bệnh phổi trong dữ liệu.
            - **So sánh tần suất:** Loại bệnh nào phổ biến nhất (cột cao nhất) và ít gặp nhất (cột thấp nhất)?
            - **Chênh lệch:** Đánh giá mức độ chênh lệch giữa các loại bệnh, có thể liên quan đến yếu tố nguy cơ hoặc dịch tễ.
            - **Lưu ý:** Mỗi bệnh nhân chỉ được ghi nhận một loại bệnh chính trong cột này.
            """)

    # Hút Thuốc & Dung Tích Phổi
    elif analysis_page == "Hút Thuốc & Dung Tích Phổi":
        st.subheader("Ảnh Hưởng của Hút Thuốc lên Dung Tích Phổi")

        # Xử lý giá trị thiếu trước khi vẽ biểu đồ
        plot_df = df.dropna(subset=["Tình Trạng Hút Thuốc", "Dung Tích Phổi"])

        # Boxplot
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        sns.boxplot(x=plot_df["Tình Trạng Hút Thuốc"], y=plot_df["Dung Tích Phổi"], ax=ax1, palette=PALETTE[:2])
        ax1.set_xticklabels(["Không Hút Thuốc", "Hút Thuốc"])
        ax1.set_xlabel("Tình Trạng Hút Thuốc")
        ax1.set_ylabel("Dung Tích Phổi (lít)")
        ax1.set_title("Dung Tích Phổi theo Tình Trạng Hút Thuốc")
        st.pyplot(fig1)

        with st.expander("Xem Nhận Xét Chi Tiết (Biểu đồ Hộp)"):
            st.markdown("""
            **Biểu đồ Hộp (Boxplot):**
            - So sánh phân bố dung tích phổi giữa người không hút thuốc và người hút thuốc (sau khi loại bỏ giá trị thiếu).
            - **Trung vị:** Đường kẻ giữa hộp cho thấy dung tích phổi trung bình. Trung vị thấp hơn ở nhóm hút thuốc gợi ý ảnh hưởng tiêu cực.
            - **IQR:** Độ cao của hộp thể hiện sự phân tán. IQR lớn hơn cho thấy dung tích phổi biến động nhiều hơn.
            - **Râu:** Độ dài râu cho thấy phạm vi dung tích phổi (không tính ngoại lai).
            - **Điểm ngoại lai:** Các điểm ngoài râu là giá trị bất thường, có thể đáng chú ý nếu tập trung ở một nhóm.
            - **Kết luận sơ bộ:** Hút thuốc có thể làm giảm dung tích phổi nếu trung vị của nhóm hút thuốc thấp hơn.
            """)

        # Scatter Plot
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        sns.scatterplot(x=plot_df["Tình Trạng Hút Thuốc"].map({"Không": 0, "Có": 1}), y=plot_df["Dung Tích Phổi"], 
                        data=plot_df, ax=ax2, alpha=0.3, color=PALETTE[0])
        ax2.set_xlabel("Tình Trạng Hút Thuốc (0: Không Hút, 1: Có Hút)")
        ax2.set_ylabel("Dung Tích Phổi (lít)")
        ax2.set_title("Dung Tích Phổi theo Tình Trạng Hút Thuốc")
        st.pyplot(fig2)

        with st.expander("Xem Nhận Xét Chi Tiết (Biểu đồ Phân Tán)"):
            st.markdown("""
            **Biểu đồ Phân Tán (Scatter Plot):**
            - Hiển thị dung tích phổi của từng bệnh nhân theo tình trạng hút thuốc (0: Không, 1: Có).
            - **Mỗi điểm:** Đại diện cho một bệnh nhân.
            - **Xu hướng:** Quan sát sự khác biệt về phân bố giữa hai nhóm.
            - **Phân tán:** Độ phân tán của điểm cho thấy mức độ biến thiên của dung tích phổi trong mỗi nhóm.
            - **Kết hợp với Boxplot:** Kết hợp hai biểu đồ để hiểu rõ hơn về ảnh hưởng của hút thuốc.
            """)

        # Kiểm định t-test thủ công
        smokers = plot_df[plot_df["Tình Trạng Hút Thuốc"] == "Có"]["Dung Tích Phổi"]
        non_smokers = plot_df[plot_df["Tình Trạng Hút Thuốc"] == "Không"]["Dung Tích Phổi"]

        if len(smokers) > 0 and len(non_smokers) > 0:
            mean_smokers = np.mean(smokers)
            mean_non_smokers = np.mean(non_smokers)
            std_smokers = np.std(smokers, ddof=1)
            std_non_smokers = np.std(non_smokers, ddof=1)
            n_smokers = len(smokers)
            n_non_smokers = len(non_smokers)
            sp = np.sqrt(((n_smokers - 1) * std_smokers**2 + (n_non_smokers - 1) * std_non_smokers**2) / (n_smokers + n_non_smokers - 2))
            t_stat = (mean_smokers - mean_non_smokers) / (sp * np.sqrt(1/n_smokers + 1/n_non_smokers))
            df_ttest = n_smokers + n_non_smokers - 2

            def z_to_p(z):
                z = abs(z)
                if z > 3.7:
                    return 0.0
                p = 1 / (1 + np.exp(0.07056 * z**3 + 1.5976 * z))
                return 2 * (1 - p)

            p_value = z_to_p(t_stat)

            st.write(f"Kiểm định t-test: t-statistic = {t_stat:.2f}, p-value = {p_value:.3f}")
            with st.expander("Giải thích kết quả t-test"):
                st.markdown(f"""
                - **t-statistic:** {t_stat:.2f} (Đo lường sự khác biệt giữa trung bình của hai nhóm).
                - **p-value:** {p_value:.3f} (Xác suất kết quả nếu không có sự khác biệt thực sự).
                - **Ý nghĩa thống kê:** 
                    - p < 0.05: Có sự khác biệt đáng kể về dung tích phổi giữa hai nhóm.
                    - p >= 0.05: Không đủ bằng chứng để kết luận có sự khác biệt.
                """)
        else:
            st.write("Không thể thực hiện kiểm định t-test: Một hoặc cả hai nhóm không có dữ liệu.")

    # Lượt Khám Bệnh
    elif analysis_page == "Lượt Khám Bệnh":
        st.subheader("Lượt Khám Bệnh Trung Bình theo Loại Bệnh")

        selected_diseases = st.multiselect("Chọn Loại Bệnh:", df["Loại Bệnh"].unique()) 
        if selected_diseases:
            # Xử lý giá trị thiếu trước khi tính trung bình
            visits_per_disease = df[df["Loại Bệnh"].isin(selected_diseases)].groupby("Loại Bệnh")["Số Lượt Khám Bệnh"].apply(lambda x: x.dropna().mean())
            st.bar_chart(visits_per_disease)

            with st.expander("Xem Nhận Xét Chi Tiết"):
                st.markdown("""
                **Biểu đồ Cột Lượt Khám Bệnh Trung Bình:**
                - Hiển thị số lượt khám bệnh trung bình cho các loại bệnh đã chọn (sau khi loại bỏ giá trị thiếu).
                - **So sánh:** So sánh chiều cao cột giữa các loại bệnh.
                - **Giá trị cụ thể:** Xác định loại bệnh nào có số lượt khám trung bình cao nhất/thấp nhất.
                - **Lưu ý:** Số lượt khám cao có thể do mức độ nghiêm trọng của bệnh hoặc yêu cầu theo dõi thường xuyên.
                - **Kết hợp:** Kết hợp với phân tích khác (ví dụ: Loại Điều Trị) để hiểu rõ hơn.
                """)
        else:
            st.write("Vui lòng chọn ít nhất một loại bệnh.")

    # Tương Quan
    elif analysis_page == "Tương Quan":
        st.subheader("Biểu Đồ Tương Quan (Heatmap)")

        # Mã hóa dữ liệu phân loại để tính tương quan
        encoded_data = encode_data(df)
        corr_matrix = encoded_data.corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
        ax.set_title("Tương Quan Giữa Các Biến Số")
        st.pyplot(fig)

        with st.expander("Xem Nhận Xét Chi Tiết"):
            st.markdown("""
            **Biểu đồ Nhiệt Tương Quan (Heatmap):**
            - Thể hiện mối tương quan tuyến tính giữa các biến số trong dữ liệu.
            - **Giải thích:**
                - Gần 1: Tương quan dương mạnh (khi một biến tăng, biến kia cũng tăng).
                - Gần -1: Tương quan âm mạnh (khi một biến tăng, biến kia giảm).
                - Gần 0: Tương quan yếu hoặc không có.
            - **Màu sắc:** Đỏ (dương), Xanh (âm), độ đậm thể hiện mức độ mạnh.
            - **Cảnh báo:** Tương quan không phải là nhân quả, chỉ phản ánh mối quan hệ tuyến tính.
            - **Cách sử dụng:** Tìm các cặp biến có tương quan mạnh (ví dụ: Tuổi và Dung Tích Phổi, Hút Thuốc và Hồi Phục).
            """)
    
    elif analysis_page == "Phân Tích Song Biến (Bivariate Analysis)":        
        # Phân tích song biến (sử dụng dữ liệu gốc)
        st.subheader("Phân Tích Song Biến")
        num_col = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        feature_x = st.selectbox("Chọn biến X:", num_col)
        feature_y = st.selectbox("Chọn biến Y:", num_col)
        plot_type = st.radio("Loại biểu đồ:", ["Scatter", "2D KDE"])

        if feature_x != feature_y:
            fig, ax = plt.subplots(figsize=(10, 6))
            if plot_type == "Scatter":
                sns.scatterplot(x=df[feature_x], y=df[feature_y], hue=df["Hồi Phục"], palette=PALETTE[:2], ax=ax)
            elif plot_type == "2D KDE":
                sns.kdeplot(x=df[feature_x], y=df[feature_y], cmap="Blues", fill=True, ax=ax)
            ax.set_title(f"{feature_x} vs {feature_y} ({plot_type})")
            st.pyplot(fig)
        else:
            st.warning("Vui lòng chọn hai biến khác nhau cho trục X và Y.")
            
        with st.expander("Một số nhận xét về các tổ hợp song biến"):    
            st.markdown("""
            1. **Mối Quan Hệ giữa Dung Tích Phổi và Tuổi:** Khi tuổi tăng, dung tích phổi có xu hướng giảm.
            2. **Tương Quan giữa Hút Thuốc và Dung Tích Phổi:** Những người hút thuốc có xu hướng có dung tích phổi thấp hơn.
            3. **Tỷ Lệ Hồi Phục và Số Lượt Khám Bệnh:** Theo dõi y tế tốt hơn giúp tăng cơ hội phục hồi.
            4. **Tác Động của Loại Bệnh lên Hồi Phục:** Bệnh mãn tính có tỷ lệ hồi phục thấp hơn.
            5. **Ảnh Hưởng của Loại Điều Trị:** Phẫu thuật hoặc liệu pháp có xu hướng có tương quan tích cực với hồi phục.
            """)

    elif analysis_page == "Tỷ lệ hồi phục":   
        # Tỷ lệ hồi phục (sử dụng dữ liệu gốc)
        st.subheader("Phân Tích Tỷ lệ hồi phục")
        factor = st.selectbox("Chọn yếu tố để so sánh Tỷ lệ hồi phục:", 
                            ["Tình Trạng Hút Thuốc", "Loại Bệnh", "Loại Điều Trị"])
        
        def rec_rate(data, factor):
            recovery_rate = data.groupby(factor)["Hồi Phục"].value_counts(normalize=True).unstack().fillna(0)
            return recovery_rate

        recovery_data = rec_rate(df, factor)
        fig, ax = plt.subplots(figsize=(10, 6))
        recovery_data.plot(kind='bar', stacked=True, color=PALETTE[:2], ax=ax)
        for p in ax.patches:
            width, height = p.get_width(), p.get_height()
            x, y = p.get_xy() 
            ax.text(x + width/2, y + height/2, f"{height:.0%}", ha='center', va='center')
        ax.set_title(f"Tỷ lệ hồi phục theo {factor}")
        ax.set_ylabel("Tỷ lệ")
        ax.legend(title="Hồi Phục", labels=["Không", "Có"])
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig)

    elif analysis_page == "Dung Lượng Phổi Trung Bình Theo Nhóm Tuổi và Loại Bệnh":
        st.subheader("Dung Lượng Phổi Trung Bình Theo Nhóm Tuổi và Loại Bệnh")

        df['Nhóm Tuổi'] = pd.cut(
                df['Tuổi'], 
                bins=[0, 20, 40, 60, 80, 100], 
                labels=['0-20', '21-40', '41-60', '61-80', '81+']
            )
            
        # Tính trung bình dung lượng phổi cho từng nhóm tuổi và loại bệnh
        lung_capacity_by_age_disease = df.groupby(['Nhóm Tuổi', 'Loại Bệnh'])['Dung Tích Phổi'].mean().unstack()
        
        # Tạo heatmap
        fig_age_lung = px.imshow(
            lung_capacity_by_age_disease, 
            title="Dung Lượng Phổi Trung Bình Theo Nhóm Tuổi và Loại Bệnh",
            labels=dict(x="Loại Bệnh", y="Nhóm Tuổi", color="Dung Lượng Phổi"),
            color_continuous_scale="YlGnBu"  # Thang màu thân thiện với người mù màu
        )
        st.plotly_chart(fig_age_lung, use_container_width=True)
        
        # Nhận xét về biểu đồ
        st.markdown("**Nhận Xét:**")
        
        # Phân tích tổng quan
        overall_analysis = lung_capacity_by_age_disease.apply(lambda x: pd.Series({
            'Nhóm Tuổi Cao Nhất': x.idxmax(),
            'Giá Trị Cao Nhất': x.max()
        }))
        
        for disease, analysis in overall_analysis.items():
            st.markdown(f"- {disease}: Dung lượng phổi cao nhất ở nhóm tuổi {analysis['Nhóm Tuổi Cao Nhất']} với giá trị {analysis['Giá Trị Cao Nhất']:.2f}")

# --- Trang 4: Nhận Xét Chung ---
elif page == "4. Nhận Xét Chung":
    st.header("4. Nhận Xét Chung")
    avg_age = df["Tuổi"].mean()
    smoking_rate = (df["Tình Trạng Hút Thuốc"] == "Có").mean() * 100
    avg_hospital_visits = df["Số Lượt Khám Bệnh"].mean()
    male_percentage = (df["Giới Tính"] == "Nam").mean() * 100
    female_percentage = (df["Giới Tính"] == "Nữ").mean() * 100

    st.markdown(f"""
    - **Tổng Quan về Dữ Liệu và Kết Quả Phân Tích:**
        - **Mối Tương Quan giữa Hút Thuốc và Dung Tích Phổi:** Dữ liệu cho thấy những bệnh nhân hút thuốc có xu hướng có dung tích phổi thấp hơn (xem boxplot).
        - **Phổ Biến của Bệnh:** Các loại bệnh phổi phổ biến nhất trong tập dữ liệu cần được xác định từ biểu đồ phân bố loại bệnh.
        - **Tuổi Trung Bình của Bệnh Nhân:** {avg_age:.2f} tuổi.
        - **Tỷ Lệ Hút Thuốc:** {smoking_rate:.2f}%.
        - **Số Lượt Khám Bệnh:** Trung bình {avg_hospital_visits:.2f} lượt.
        - **Giới Tính:** Nam: {male_percentage:.2f}%, Nữ: {female_percentage:.2f}%.
    """)

    st.subheader("Hạn Chế")
    st.markdown("""
    - **Kích Thước Mẫu:** Kích thước mẫu có thể không đủ lớn để đưa ra kết luận chắc chắn.
    - **Thiếu Sót Dữ Liệu:** Một số hàng có thể thiếu giá trị (nếu có).
    - **Đơn Vị Đo:** Dung tích phổi được đo bằng lít, không chuẩn hóa.
    """)