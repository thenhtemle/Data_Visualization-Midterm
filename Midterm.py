import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sklearn.preprocessing as skp
import plotly.express as px
from matplotlib.backends.backend_pdf import PdfPages
from io import BytesIO

# Thiết lập cấu hình trang
st.set_page_config(page_title="Phân Tích Bệnh Phổi", page_icon="🫁", layout="wide", initial_sidebar_state="expanded")

# Thiết lập bảng màu
sns.set_palette("colorblind")
PALETTE = ["#88CCEE", "#44AA99", "#117733", "#999933", "#DDCC77", "#CC6677", "#882255", "#AA4499"]

# --- Hàm tạo PDF từ figure ---
def save_fig_to_pdf(fig):
    buffer = BytesIO()
    with PdfPages(buffer) as pdf:
        pdf.savefig(fig, bbox_inches='tight')
    buffer.seek(0)
    return buffer

# --- Hàm tải và tiền xử lý dữ liệu ---
@st.cache_data(show_spinner=False)
def load_and_preprocess_data(file):
    try:
        df = pd.read_csv(file, na_values=["None", " ", "UNKNOWN", -1, 999, "NA", "N/A", "NULL"])
        col_mapping = {
            "Age": "Tuổi", "Gender": "Giới Tính", "Smoking Status": "Tình Trạng Hút Thuốc",
            "Lung Capacity": "Dung Tích Phổi", "Disease Type": "Loại Bệnh", 
            "Treatment Type": "Loại Điều Trị", "Hospital Visits": "Số Lượt Khám Bệnh", 
            "Recovered": "Hồi Phục"
        }
        df.columns = [col_mapping.get(col, col) for col in df.columns]
        numeric_cols = ["Tuổi", "Dung Tích Phổi", "Số Lượt Khám Bệnh"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df["Hồi Phục"] = df["Hồi Phục"].map({"Có": 1, "Yes": 1, "Không": 0, "No": 0})
        categorical_cols = ["Giới Tính", "Tình Trạng Hút Thuốc", "Loại Bệnh", "Loại Điều Trị"]
        for col in categorical_cols:
            df[col] = df[col].astype('category')
        return df
    except Exception as e:
        st.error(f"Lỗi khi xử lý dữ liệu: {e}")
        st.stop()

# Hàm mã hóa dữ liệu phân loại
def encode_data(data):
    encoder = skp.LabelEncoder()
    encoded_data = data.copy()
    for col in data.select_dtypes(include=['object', 'category']).columns:
        encoded_data[col] = encoder.fit_transform(data[col].astype(str))
    return encoded_data

# Hàm vẽ đồ thị tỷ lệ phục hồi bệnh
def plot_recovery_by_disease(df, chart_type='Pie'):
    recovery_rates = df.groupby('Loại Bệnh')['Hồi Phục'].value_counts(normalize=True).unstack() * 100
    
    if chart_type == 'Stacked':
        plt.figure(figsize=(10, 6))
        recovery_rates.plot(kind='bar', stacked=True, color=['#FF6B6B', '#4ECDC4'])
        plt.title('Tỷ lệ phục hồi theo loại bệnh', fontsize=14, pad=15)
        plt.xlabel('Loại bệnh', fontsize=12)
        plt.ylabel('Tỷ lệ (%)', fontsize=12)
        plt.legend(title='Hồi phục', loc='center left', bbox_to_anchor=(1.0, 0.5))
        plt.xticks(rotation=45, ha='right')
        plt.yticks(ticks=range(0, 101, 10), labels=[f"{i}%" for i in range(0, 101, 10)])
        plt.tight_layout()
    else:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
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

# Hàm tạo biểu đồ ảnh hưởng của hút thuốc
def plot_smoking_impact(df, chart_type='Stacked'):
    smoking_impact = df.groupby('Tình Trạng Hút Thuốc')['Hồi Phục'].value_counts(normalize=True).unstack() * 100
    
    if chart_type == 'Stacked':
        plt.figure(figsize=(8, 6))
        smoking_impact.plot(kind='bar', stacked=True, color=['#FF6B6B', '#4ECDC4'])
        plt.title('Ảnh hưởng của hút thuốc đến khả năng phục hồi', fontsize=14, pad=15)
        plt.xlabel('Tình trạng hút thuốc', fontsize=12)
        plt.ylabel('Tỷ lệ (%)', fontsize=12)
        plt.legend(title='Phục Hồi', loc='center left', bbox_to_anchor=(1.0, 0.5))
        plt.xticks(rotation=0)
        plt.yticks(ticks=range(0, 101, 10), labels=[f"{i}%" for i in range(0, 101, 10)])
        plt.tight_layout()
    else:
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

# --- CSS tùy chỉnh ---
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
        pdf_buffer = save_fig_to_pdf(fig)
        st.download_button(
            label="Lưu biểu đồ dưới dạng PDF",
            data=pdf_buffer,
            file_name=f"PhanBo_{num_select}.pdf",
            mime="application/pdf"
        )

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
        pdf_buffer = save_fig_to_pdf(fig)
        st.download_button(
            label="Lưu biểu đồ dưới dạng PDF",
            data=pdf_buffer,
            file_name=f"PhanBo_{cat_select}.pdf",
            mime="application/pdf"
        )

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
    
    if analysis_page == "Thống kê chung":
        st.subheader("Thống Kê Chung")
        total_patients = len(df)
        avg_age = df["Tuổi"].mean()
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

    elif analysis_page == "Tuổi & Dung Tích Phổi":
        st.subheader("Phân Bố Tuổi & Dung Tích Phổi")
        filtered_df = df.dropna(subset=["Tuổi", "Dung Tích Phổi"])
        if not filtered_df["Tuổi"].isnull().all():
            age_range = st.slider("Chọn Khoảng Tuổi", int(filtered_df["Tuổi"].min()), int(filtered_df["Tuổi"].max()), 
                                  (int(filtered_df["Tuổi"].min()), int(filtered_df["Tuổi"].max())))
            filtered_df = filtered_df[(filtered_df["Tuổi"] >= age_range[0]) & (filtered_df["Tuổi"] <= age_range[1])]
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
            pdf_buffer = save_fig_to_pdf(fig)
            st.download_button(
                label="Lưu biểu đồ dưới dạng PDF",
                data=pdf_buffer,
                file_name="PhanBo_Tuoi_DungTichPhoi.pdf",
                mime="application/pdf"
            )
        else:
            st.warning("Không có dữ liệu tuổi hợp lệ để hiển thị.")

    elif analysis_page == "Loại Bệnh":
        st.subheader("Phân Bố Loại Bệnh")
        disease_counts = df["Loại Bệnh"].value_counts()
        fig = plt.figure(figsize=(10, 6))
        disease_counts.plot(kind='bar')
        plt.title("Phân Bố Loại Bệnh")
        plt.xlabel("Loại Bệnh")
        plt.ylabel("Số Lượng")
        st.pyplot(fig)
        pdf_buffer = save_fig_to_pdf(fig)
        st.download_button(
            label="Lưu biểu đồ dưới dạng PDF",
            data=pdf_buffer,
            file_name="PhanBo_LoaiBenh.pdf",
            mime="application/pdf"
        )

    elif analysis_page == "Hút Thuốc & Dung Tích Phổi":
        st.subheader("Ảnh Hưởng của Hút Thuốc lên Dung Tích Phổi")
        plot_df = df.dropna(subset=["Tình Trạng Hút Thuốc", "Dung Tích Phổi"])
        
        # Boxplot
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        sns.boxplot(x=plot_df["Tình Trạng Hút Thuốc"], y=plot_df["Dung Tích Phổi"], ax=ax1, palette=PALETTE[:2])
        ax1.set_xticklabels(["Không Hút Thuốc", "Hút Thuốc"])
        ax1.set_xlabel("Tình Trạng Hút Thuốc")
        ax1.set_ylabel("Dung Tích Phổi (lít)")
        ax1.set_title("Dung Tích Phổi theo Tình Trạng Hút Thuốc")
        st.pyplot(fig1)
        pdf_buffer1 = save_fig_to_pdf(fig1)
        st.download_button(
            label="Lưu biểu đồ dưới dạng PDF",
            data=pdf_buffer1,
            file_name="DungTichPhoi_Theo_HutThuoc_Boxplot.pdf",
            mime="application/pdf"
        )

        # Scatter Plot
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        sns.scatterplot(x=plot_df["Tình Trạng Hút Thuốc"].map({"Không": 0, "Có": 1}), y=plot_df["Dung Tích Phổi"], 
                        data=plot_df, ax=ax2, alpha=0.3, color=PALETTE[0])
        ax2.set_xlabel("Tình Trạng Hút Thuốc (0: Không Hút, 1: Có Hút)")
        ax2.set_ylabel("Dung Tích Phổi (lít)")
        ax2.set_title("Dung Tích Phổi theo Tình Trạng Hút Thuốc")
        st.pyplot(fig2)
        pdf_buffer2 = save_fig_to_pdf(fig2)
        st.download_button(
            label="Lưu biểu đồ dưới dạng PDF",
            data=pdf_buffer2,
            file_name="DungTichPhoi_Theo_HutThuoc_Scatter.pdf",
            mime="application/pdf"
        )

    elif analysis_page == "Lượt Khám Bệnh":
        st.subheader("Lượt Khám Bệnh Trung Bình theo Loại Bệnh")
        selected_diseases = st.multiselect("Chọn Loại Bệnh:", df["Loại Bệnh"].unique()) 
        if selected_diseases:
            visits_per_disease = df[df["Loại Bệnh"].isin(selected_diseases)].groupby("Loại Bệnh")["Số Lượt Khám Bệnh"].apply(lambda x: x.dropna().mean())
            fig = plt.figure(figsize=(10, 6))
            visits_per_disease.plot(kind='bar')
            plt.title("Lượt Khám Bệnh Trung Bình theo Loại Bệnh")
            plt.xlabel("Loại Bệnh")
            plt.ylabel("Số Lượt Khám Trung Bình")
            st.pyplot(fig)
            pdf_buffer = save_fig_to_pdf(fig)
            st.download_button(
                label="Lưu biểu đồ dưới dạng PDF",
                data=pdf_buffer,
                file_name="LuotKhamBenh_Theo_LoaiBenh.pdf",
                mime="application/pdf"
            )

    elif analysis_page == "Tương Quan":
        st.subheader("Biểu Đồ Tương Quan (Heatmap)")
        encoded_data = encode_data(df)
        corr_matrix = encoded_data.corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
        ax.set_title("Tương Quan Giữa Các Biến Số")
        st.pyplot(fig)
        pdf_buffer = save_fig_to_pdf(fig)
        st.download_button(
            label="Lưu biểu đồ dưới dạng PDF",
            data=pdf_buffer,
            file_name="TuongQuan.pdf",
            mime="application/pdf"
        )

    elif analysis_page == "Phân Tích Song Biến (Bivariate Analysis)":        
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
            pdf_buffer = save_fig_to_pdf(fig)
            st.download_button(
                label="Lưu biểu đồ dưới dạng PDF",
                data=pdf_buffer,
                file_name=f"{feature_x}_vs_{feature_y}_{plot_type}.pdf",
                mime="application/pdf"
            )

    elif analysis_page == "Tỷ lệ hồi phục":   
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
        pdf_buffer = save_fig_to_pdf(fig)
        st.download_button(
            label="Lưu biểu đồ dưới dạng PDF",
            data=pdf_buffer,
            file_name=f"TyLeHoiPhuc_Theo_{factor}.pdf",
            mime="application/pdf"
        )

    elif analysis_page == "Dung Lượng Phổi Trung Bình Theo Nhóm Tuổi và Loại Bệnh":
        st.subheader("Dung Lượng Phổi Trung Bình Theo Nhóm Tuổi và Loại Bệnh")
        df['Nhóm Tuổi'] = pd.cut(df['Tuổi'], bins=[0, 20, 40, 60, 80, 100], labels=['0-20', '21-40', '41-60', '61-80', '81+'])
        lung_capacity_by_age_disease = df.groupby(['Nhóm Tuổi', 'Loại Bệnh'])['Dung Tích Phổi'].mean().unstack()
        fig_age_lung = px.imshow(
            lung_capacity_by_age_disease, 
            title="Dung Lượng Phổi Trung Bình Theo Nhóm Tuổi và Loại Bệnh",
            labels=dict(x="Loại Bệnh", y="Nhóm Tuổi", color="Dung Lượng Phổi"),
            color_continuous_scale="YlGnBu"
        )
        st.plotly_chart(fig_age_lung, use_container_width=True)
        # Note: Plotly charts need a different approach for PDF export; skipping here for simplicity

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