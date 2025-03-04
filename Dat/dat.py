import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import numpy as np

# Cấu hình trang
st.set_page_config(page_title="Phân Tích Bệnh Phổi", page_icon="🫁", layout="wide")

# Hàm tải dữ liệu
@st.cache_data
def load_data():
    return pd.read_csv('lung_disease_data_cleaned.csv')

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

# Hàm chính tạo bảng điều khiển
def main():
    # Tiêu đề và giới thiệu
    st.title("🫁 Bảng Phân Tích Dữ Liệu Bệnh Phổi")
    st.write("Phân tích toàn diện về dữ liệu bệnh nhân bệnh phổi")

    # Tải dữ liệu
    df = load_data()

    # Thanh bên để lọc
    st.sidebar.header("🔍 Bộ Lọc")
    
    # Lọc theo Giới Tính
    gender_filter = st.sidebar.multiselect(
        "Chọn Giới Tính",
        options=df['Giới Tính'].unique(),
        default=df['Giới Tính'].unique()
    )

    # Lọc theo Tình Trạng Hút Thuốc
    smoking_statuses = df['Tình Trạng Hút Thuốc'].unique()
    
    # Kiểm tra nếu có dữ liệu về hút thuốc
    if len(smoking_statuses) > 0:
        smoking_filter = st.sidebar.multiselect(
            "Chọn Tình Trạng Hút Thuốc",
            options=smoking_statuses,
            default=smoking_statuses
        )
    else:
        # Nếu không có dữ liệu về hút thuốc, bỏ qua bộ lọc này
        smoking_filter = smoking_statuses

    # Lọc theo Loại Bệnh
    disease_filter = st.sidebar.multiselect(
        "Chọn Loại Bệnh",
        options=df['Loại Bệnh'].unique(),
        default=df['Loại Bệnh'].unique()
    )

    # Áp dụng bộ lọc
    filtered_df = df[
        (df['Giới Tính'].isin(gender_filter)) & 
        (len(smoking_filter) == 0 or df['Tình Trạng Hút Thuốc'].isin(smoking_filter)) & 
        (df['Loại Bệnh'].isin(disease_filter))
    ]

    # Thêm thông báo nếu không có dữ liệu về hút thuốc
    if len(smoking_statuses) == 0:
        st.warning("⚠️ Không có dữ liệu về tình trạng hút thuốc trong tập dữ liệu.")

    # Tạo các tab
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Tổng Quan", 
        "🌈 Phân Bổ Bệnh", 
        "📈 Phân Tích Dung Lượng Phổi", 
        "🏥 Kết Quả Điều Trị"
    ])

    with tab1:
        # Các chỉ số Tổng Quan
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_patients = len(filtered_df)
            st.metric("Tổng Số Bệnh Nhân", total_patients)
            st.caption(f"Chiếm {total_patients/len(df)*100:.2f}% tổng số bệnh nhân")
        
        with col2:
            recovery_rate = filtered_df['Phục Hồi'].value_counts(normalize=True).get('Có', 0)*100
            st.metric("Tỷ Lệ Phục Hồi", f"{recovery_rate:.2f}%")
            st.caption("Tỷ lệ bệnh nhân được điều trị thành công")
        
        with col3:
            avg_hospital_visits = filtered_df['Số Lần Khám'].mean()
            st.metric("Trung Bình Lần Khám", f"{avg_hospital_visits:.2f}")
            st.caption("Số lần khám trung bình của bệnh nhân")
        
        with col4:
            avg_age = filtered_df['Tuổi'].mean()
            st.metric("Tuổi Trung Bình", f"{avg_age:.2f}")
            st.caption("Tuổi trung bình của nhóm bệnh nhân")

    with tab2:
        # Phân Bổ Bệnh
        col1, col2 = st.columns(2)
        
        with col1:
            # Biểu đồ tròn về Loại Bệnh
            disease_counts = filtered_df['Loại Bệnh'].value_counts()
            fig_disease = px.pie(
                disease_counts, 
                values=disease_counts.values, 
                names=disease_counts.index, 
                title="Phân Bổ Các Loại Bệnh Phổi",
                color_discrete_sequence=COLOR_PALETTE
            )
            st.plotly_chart(fig_disease, use_container_width=True)
            
            # Nhận xét về biểu đồ
            st.markdown("**Nhận Xét:**")
            for disease, count in disease_counts.items():
                percentage = count / len(filtered_df) * 100
                st.markdown(f"- {disease}: {count} bệnh nhân ({percentage:.2f}%)")
        
        with col2:
            # Kiểm tra xem có dữ liệu về tình trạng hút thuốc không
            smoking_statuses = df['Tình Trạng Hút Thuốc'].unique()
            
            if len(smoking_statuses) > 0:
                # Biểu đồ cột về Loại Bệnh theo Tình Trạng Hút Thuốc
                disease_smoking = filtered_df.groupby(['Loại Bệnh', 'Tình Trạng Hút Thuốc']).size().unstack(fill_value=0)
                fig_disease_smoking = px.bar(
                    disease_smoking, 
                    title="Các Bệnh Phổi Theo Tình Trạng Hút Thuốc",
                    labels={'value': 'Số Bệnh Nhân', 'Tình Trạng Hút Thuốc': 'Tình Trạng Hút Thuốc'},
                    color_discrete_sequence=COLOR_PALETTE
                )
                st.plotly_chart(fig_disease_smoking, use_container_width=True)
                
                # Nhận xét về biểu đồ
                st.markdown("**Nhận Xét:**")
                for disease in disease_smoking.index:
                    status_details = []
                    total_for_disease = disease_smoking.loc[disease].sum()
                    
                    for status in disease_smoking.columns:
                        count = disease_smoking.loc[disease, status]
                        percent = count / total_for_disease * 100
                        status_details.append(f"{status}: {count} người ({percent:.2f}%)")
                    
                    st.markdown(f"- {disease}: " + ", ".join(status_details))
            else:
                st.info("💡 Không có dữ liệu để hiển thị về tình trạng hút thuốc.")

    with tab3:
        # Phân Tích Dung Lượng Phổi
        col1, col2 = st.columns(2)
        
        with col1:
            # Biểu đồ hộp về Dung Lượng Phổi theo Loại Bệnh
            fig_lung_capacity = px.box(
                filtered_df, 
                x='Loại Bệnh', 
                y='Dung Lượng Phổi', 
                title="Phân Bố Dung Lượng Phổi Theo Loại Bệnh",
                color='Loại Bệnh',
                color_discrete_sequence=COLOR_PALETTE
            )
            st.plotly_chart(fig_lung_capacity, use_container_width=True)
            
            # Nhận xét về biểu đồ
            st.markdown("**Nhận Xét:**")
            for disease in filtered_df['Loại Bệnh'].unique():
                disease_data = filtered_df[filtered_df['Loại Bệnh'] == disease]['Dung Lượng Phổi']
                st.markdown(f"- {disease}: Trung bình {disease_data.mean():.2f}, Dao động từ {disease_data.min():.2f} đến {disease_data.max():.2f}")
        
        with col2:
            # Biểu đồ heatmap của Tuổi và Dung Lượng Phổi
            # Chia tuổi và dung lượng phổi thành các nhóm
            filtered_df['Nhóm Tuổi'] = pd.cut(
                filtered_df['Tuổi'], 
                bins=[0, 20, 40, 60, 80, 100], 
                labels=['0-20', '21-40', '41-60', '61-80', '81+']
            )
            
            # Tính trung bình dung lượng phổi cho từng nhóm tuổi và loại bệnh
            lung_capacity_by_age_disease = filtered_df.groupby(['Nhóm Tuổi', 'Loại Bệnh'])['Dung Lượng Phổi'].mean().unstack()
            
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
            

    with tab4:
        # Kết Quả Điều Trị
        col1, col2 = st.columns(2)
        
        with col1:
            # Tỷ lệ phục hồi theo Loại Điều Trị
            treatment_recovery = filtered_df.groupby(['Loại Điều Trị', 'Phục Hồi']).size().unstack(fill_value=0)
            treatment_recovery_pct = treatment_recovery.div(treatment_recovery.sum(axis=1), axis=0) * 100
            
            fig_treatment_recovery = px.bar(
                treatment_recovery_pct, 
                title="Tỷ Lệ Phục Hồi Theo Loại Điều Trị",
                labels={'value': 'Phần Trăm', 'Phục Hồi': 'Trạng Thái Phục Hồi'},
                color_discrete_sequence=COLOR_PALETTE
            )
            st.plotly_chart(fig_treatment_recovery, use_container_width=True)
            
            # Nhận xét về biểu đồ
            st.markdown("**Nhận Xét:**")
            for treatment in treatment_recovery_pct.index:
                recovery_rate = treatment_recovery_pct.loc[treatment, 'Có']
                st.markdown(f"- {treatment}: Tỷ lệ phục hồi {recovery_rate:.2f}%")
        
        with col2:
            # Số lần khám bệnh theo Trạng Thái Phục Hồi
            fig_visits_recovery = px.box(
                filtered_df, 
                x='Phục Hồi', 
                y='Số Lần Khám', 
                title="Số Lần Khám Bệnh Theo Trạng Thái Phục Hồi",
                color='Phục Hồi',
                color_discrete_sequence=[COLOR_PALETTE[2], COLOR_PALETTE[3]]
            )
            st.plotly_chart(fig_visits_recovery, use_container_width=True)
            
            # Nhận xét về biểu đồ
            recovered_visits = filtered_df[filtered_df['Phục Hồi'] == 'Có']['Số Lần Khám']
            not_recovered_visits = filtered_df[filtered_df['Phục Hồi'] == 'Không']['Số Lần Khám']
            
            st.markdown("**Nhận Xét:**")
            st.markdown(f"- Bệnh nhân khỏi bệnh: Trung bình {recovered_visits.mean():.2f} lần khám")
            st.markdown(f"- Bệnh nhân chưa khỏi: Trung bình {not_recovered_visits.mean():.2f} lần khám")

    # Trình xem dữ liệu thô
    if st.checkbox("Xem Dữ Liệu Gốc"):
        st.dataframe(filtered_df)

# Chạy ứng dụng
if __name__ == "__main__":
    main()

# Yêu cầu (tương đương requirements.txt):
# streamlit
# pandas
# plotly