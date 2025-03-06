import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# Cấu hình trang
st.set_page_config(page_title="Phân Tích Bệnh Phổi", page_icon="🫁", layout="wide")

# Bảng màu thân thiện với người mù màu
COLOR_PALETTE =  [
    '#1F77B4',  # Xanh dương
    '#FF7F0E',  # Cam
    '#2CA02C',  # Xanh lá
    '#D62728',  # Đỏ
    '#9467BD',  # Tím
    '#8C564B',  # Nâu
    '#E377C2',  # Hồng
    '#7F7F7F'   # Xám
]

# Hàm tải dữ liệu
@st.cache_data
def load_data():
    try:
        data = pd.read_csv('lung_disease_data_preprocessed.csv')
        return data
    except FileNotFoundError:
        st.error("Không tìm thấy file lung_disease_data_preprocessed.csv trong thư mục Data. Hãy đảm bảo bạn đã chạy code tiền xử lý trước.")
        st.stop()

# Hàm chính tạo bảng điều khiển
def main():
    # Tiêu đề và giới thiệu
    st.title("🫁 Dashboard Bệnh Phổi")
    st.write("Phân tích toàn diện về dữ liệu bệnh nhân bệnh phổi")

    # Tải dữ liệu
    df = load_data()

    # Đổi tên cột (từ tiếng Anh sang tiếng Việt cho trực quan)
    df = df.rename(columns={
        'Age': 'Tuổi',
        'Lung Capacity': 'Dung Lượng Phổi',
        'Hospital Visits': 'Số Lần Khám',
        'Recovered': 'Phục Hồi'
    })

    # Sidebar để lọc
    st.sidebar.header("🔍 Bộ Lọc")

    # Lọc theo Giới Tính
    gender_options = [col.replace('Gender_', '') for col in df.columns if 'Gender_' in col]
    gender_filter = st.sidebar.multiselect(
        "Chọn Giới Tính",
        options=gender_options,
        default=gender_options
    )

    # Lọc theo Tình Trạng Hút Thuốc
    smoking_options = [col.replace('Smoking Status_', '') for col in df.columns if 'Smoking Status_' in col]
    smoking_filter = st.sidebar.multiselect(
        "Chọn Tình Trạng Hút Thuốc",
        options=smoking_options,
        default=smoking_options
    )

    # Lọc theo Loại Bệnh
    disease_options = [col.replace('Disease Type_', '') for col in df.columns if 'Disease Type_' in col]
    disease_filter = st.sidebar.multiselect(
        "Chọn Loại Bệnh",
        options=disease_options,
        default=disease_options
    )

    # Lọc theo Loại Điều Trị
    treatment_options = [col.replace('Treatment Type_', '') for col in df.columns if 'Treatment Type_' in col]
    treatment_filter = st.sidebar.multiselect(
        "Chọn Loại Điều Trị",
        options=treatment_options,
        default=treatment_options
    )

    # Áp dụng bộ lọc
    def filter_data(df, gender, smoking, disease, treatment):
        filtered_df = df.copy()

        def safe_filter(df, col_prefix, selected_options):
            valid_cols = [col for col in df.columns if col.startswith(col_prefix) and col.split('_')[-1] in selected_options]
            if valid_cols:
                return df[valid_cols].any(axis=1)
            return pd.Series([True] * len(df), index=df.index)

        gender_filter_series = safe_filter(filtered_df, "Gender_", gender)
        smoking_filter_series = safe_filter(filtered_df, "Smoking Status_", smoking)
        disease_filter_series = safe_filter(filtered_df, "Disease Type_", disease)
        treatment_filter_series = safe_filter(filtered_df, "Treatment Type_", treatment)

        combined_filter = gender_filter_series & smoking_filter_series & disease_filter_series & treatment_filter_series
        return filtered_df[combined_filter]

    filtered_df = filter_data(df, gender_filter, smoking_filter, disease_filter, treatment_filter)

    # Thêm thông báo nếu không có dữ liệu
    if len(filtered_df) == 0:
        st.warning("⚠️ Không có dữ liệu nào khớp với bộ lọc đã chọn.")
        st.stop()

    # Các tab
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
            recovery_rate = filtered_df['Phục Hồi'].value_counts(normalize=True).get(1, 0)*100
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
            disease_cols = [col for col in filtered_df.columns if 'Disease Type_' in col]
            disease_names = [col.replace('Disease Type_', '') for col in disease_cols]
            disease_counts = filtered_df[disease_cols].sum()
            fig_disease = px.pie(
                names=disease_names,
                values=disease_counts.values,
                title="Phân Bổ Các Loại Bệnh Phổi",
                color_discrete_sequence=COLOR_PALETTE
            )
            st.plotly_chart(fig_disease, use_container_width=True)

            # Nhận xét về biểu đồ
            st.markdown("**Nhận Xét:**")
            for disease, count in zip(disease_names, disease_counts):
                percentage = count / len(filtered_df) * 100
                st.markdown(f"- {disease}: {count} bệnh nhân ({percentage:.2f}%)")

        with col2:
            # Tỷ lệ hút thuốc theo bệnh
            smoking_cols = [col for col in filtered_df.columns if 'Smoking Status_' in col]
            smoking_names = [col.replace('Smoking Status_', '') for col in smoking_cols]

            if smoking_cols:
                # Tạo DataFrame mới với tổng số bệnh nhân theo từng tình trạng hút thuốc cho mỗi loại bệnh
                smoking_by_disease = filtered_df[[col for col in filtered_df.columns if 'Disease Type_' in col] + [col for col in filtered_df.columns if 'Smoking Status_' in col]].groupby([col for col in filtered_df.columns if 'Disease Type_' in col]).sum()
                smoking_by_disease.columns = [col.replace('Smoking Status_', '') for col in smoking_by_disease.columns]
                fig_smoking = px.imshow(smoking_by_disease.T,
                                title="Mối quan hệ giữa Tình Trạng Hút Thuốc và Loại Bệnh",
                                labels=dict(x="Loại Bệnh", y="Tình Trạng Hút Thuốc"),
                                color_continuous_scale="RdBu")
                st.plotly_chart(fig_smoking, use_container_width=True)
                st.markdown("**Nhận Xét:**")
            else:
                st.info("💡 Không có dữ liệu để hiển thị về tình trạng hút thuốc.")

    with tab3:
    # Phân Tích Dung Lượng Phổi
        col1, col2 = st.columns(2)

        with col1:
            # Chuẩn bị dữ liệu để vẽ biểu đồ boxplot
            filtered_df_no_na = filtered_df.dropna(subset=['Dung Lượng Phổi'])
            disease_cols = [col for col in filtered_df_no_na.columns if 'Disease Type_' in col]
            disease_names = [col.replace('Disease Type_', '') for col in disease_cols]

            # Tạo danh sách để lưu dữ liệu
            data_for_boxplot = []
            for disease_col, disease_name in zip(disease_cols, disease_names):
                data_for_boxplot.extend([(disease_name, lung_capacity) for lung_capacity in filtered_df_no_na[filtered_df_no_na[disease_col] == 1]['Dung Lượng Phổi']])
            df_boxplot = pd.DataFrame(data_for_boxplot, columns=['Loại bệnh', 'Dung Lượng Phổi'])

            fig_lung_capacity = px.box(
                df_boxplot,
                x='Loại bệnh',
                y='Dung Lượng Phổi',
                title="Phân Bố Dung Lượng Phổi Theo Loại Bệnh",
                color='Loại bệnh',
                color_discrete_sequence=COLOR_PALETTE
            )
            st.plotly_chart(fig_lung_capacity, use_container_width=True)

            # Nhận xét về biểu đồ
            st.markdown("**Nhận Xét:**")
            for disease in disease_names:
                disease_data = df_boxplot[df_boxplot['Loại bệnh'] == disease]['Dung Lượng Phổi']
                if not disease_data.empty:
                    st.markdown(f"- {disease}: Trung bình {disease_data.mean():.2f}, Dao động từ {disease_data.min():.2f} đến {disease_data.max():.2f}")
                else:
                    st.markdown(f"- Không có dữ liệu cho bệnh {disease}")
        with col2:
            # Biểu đồ heatmap của Tuổi và Dung Lượng Phổi
            # Chia tuổi và dung lượng phổi thành các nhóm
            bins = [0, 20, 40, 60, 80, 100]
            labels = ['0-20', '21-40', '41-60', '61-80', '81+']

            # Tạo cột Nhóm Tuổi
            if 'Nhóm Tuổi' in filtered_df.columns:
                filtered_df.drop(columns=['Nhóm Tuổi'], inplace=True)
            filtered_df['Nhóm Tuổi'] = pd.cut(filtered_df['Tuổi'], bins=bins, labels=labels)

            # Tính trung bình dung lượng phổi cho từng nhóm tuổi và loại bệnh
            lung_capacity_by_age_disease = filtered_df.groupby(['Nhóm Tuổi'])['Dung Lượng Phổi'].mean()

            # Tạo cột Nhóm Tuổi
            fig_age_lung = px.bar(lung_capacity_by_age_disease,
                title="Dung Lượng Phổi Trung Bình Theo Nhóm Tuổi",
                labels=dict(value="Dung Lượng Phổi", index="Nhóm Tuổi", color="Nhóm Tuổi"),
                color = lung_capacity_by_age_disease.index, # Đặt màu theo tên nhóm tuổi
                color_discrete_sequence=px.colors.sequential.YlGnBu
            )
            st.plotly_chart(fig_age_lung, use_container_width=True)

            # Nhận xét về biểu đồ
            st.markdown("**Nhận Xét:**")

            overall_analysis = pd.Series({
                'Nhóm Tuổi Cao Nhất': lung_capacity_by_age_disease.idxmax(),
                'Giá Trị Cao Nhất': lung_capacity_by_age_disease.max()
            })
            st.markdown(f"- Dung lượng phổi cao nhất ở nhóm tuổi {overall_analysis['Nhóm Tuổi Cao Nhất']} với giá trị {overall_analysis['Giá Trị Cao Nhất']:.2f}")

    with tab4:
        # Kết Quả Điều Trị
        col1, col2 = st.columns(2)

        with col1:
            # Tỷ lệ phục hồi theo Loại Điều Trị
            treatment_cols = [col for col in filtered_df.columns if 'Treatment Type_' in col]
            if treatment_cols:
                treatment_recovery_data = {}
                treatment_names = [col.replace('Treatment Type_', '') for col in treatment_cols]

                for treatment in treatment_names:
                    treatment_recovery_data[treatment] = {'Phục Hồi': len(filtered_df[(filtered_df[f"Treatment Type_{treatment}"] == 1) & (filtered_df["Phục Hồi"] == 1)]),
                                                            'Không': len(filtered_df[(filtered_df[f"Treatment Type_{treatment}"] == 1) & (filtered_df["Phục Hồi"] == 0)])
                                                            }

                treatment_recovery = pd.DataFrame(treatment_recovery_data).T
                #Check nan and fillna(0)
                treatment_recovery = treatment_recovery.fillna(0)
                treatment_recovery_pct = treatment_recovery.div(treatment_recovery.sum(axis=1), axis=0) * 100
                treatment_recovery_pct = treatment_recovery_pct.fillna(0) # Sau khi div có thể có nan vì chia 0
                fig_treatment_recovery = px.bar(
                    treatment_recovery_pct,
                    title="Tỷ Lệ Phục Hồi Theo Loại Điều Trị",
                    labels={'value': 'Phần Trăm', 'variable': 'Trạng Thái Phục Hồi'},
                    color_discrete_sequence=COLOR_PALETTE
                )
                st.plotly_chart(fig_treatment_recovery, use_container_width=True)

                # Nhận xét về biểu đồ
                st.markdown("**Nhận Xét:**")
                for treatment in treatment_recovery_pct.index:
                    if 'Phục Hồi' in treatment_recovery_pct.columns:
                        recovery_rate = treatment_recovery_pct.loc[treatment, 'Phục Hồi']
                    else:
                        recovery_rate = 0
                    st.markdown(f"- {treatment}: Tỷ lệ phục hồi {recovery_rate:.2f}%")
            else:
                st.info("⚠️ Không có dữ liệu về loại điều trị.")

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
            recovered_visits = filtered_df[filtered_df['Phục Hồi'] == 1]['Số Lần Khám']
            not_recovered_visits = filtered_df[filtered_df['Phục Hồi'] == 0]['Số Lần Khám']

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