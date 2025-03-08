import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set color palette
sns.set_palette("colorblind")

# --- 1. Data Loading and Preprocessing (Reusable Function) ---
def load_and_preprocess_data(file_path):
    try:
        df = pd.read_csv(file_path, na_values=["None", " ", "UNKNOWN", -1, 999, "NA", "N/A", "NULL", ""])
        df["Age"] = pd.to_numeric(df["Age"], errors='coerce')
        df["Lung Capacity"] = pd.to_numeric(df["Lung Capacity"], errors='coerce')
        df["Hospital Visits"] = pd.to_numeric(df["Hospital Visits"], errors='coerce')
        df["Recovered"] = pd.to_numeric(df["Recovered"], errors='coerce')
        return df
    except FileNotFoundError:
        st.error(f"Lỗi: Không tìm thấy tệp tại {file_path}.")
        st.stop()
    except Exception as e:
        st.error(f"Lỗi: {e}")
        st.stop()



# --- Load Data ---
file_path = "lung_disease_data_preprocessed.csv"
df = load_and_preprocess_data(file_path)
disease_cols = [col for col in df.columns if "Disease Type" in col]

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

    /* Xóa dấu chấm */
    div[role="radiogroup"] label span {
        visibility: hidden;
        width: 0;
        margin: 0;
        padding: 0;
    }

    /* Hiệu ứng hover */
    div[role="radiogroup"] label:hover {
        background-color: #FF851B;
        color: white !important;
        transform: scale(1.02);
    }

    /* Khi được chọn */
    div[role="radiogroup"] label[data-selected="true"] {
        background-color: #FF851B !important;
        color: white !important;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# --- Streamlit App ---
st.title("🫁 Bảng Điều Khiển Bệnh Phổi")

# --- Sidebar Navigation ---
page = st.sidebar.radio("Chọn Trang", ["1. Giới Thiệu Dữ Liệu", "2. Thống Kê Mô Tả", "3. Phân Tích Chuyên Sâu", "4. Nhận Xét Chung"], index=0)

# --- Page 1: Data Introduction ---
if page == "1. Giới Thiệu Dữ Liệu":
    st.header("1. Giới Thiệu Dữ Liệu")

    st.subheader("Nguồn Gốc Dữ Liệu")
    st.markdown("""
    - **(Thay thế phần này bằng thông tin thực tế về nguồn gốc dữ liệu của bạn)**
    - Ví dụ:
        - Dữ liệu được thu thập từ một nghiên cứu về bệnh phổi tại [Tên bệnh viện/trung tâm nghiên cứu].
        - Dữ liệu bao gồm thông tin của [Số lượng] bệnh nhân trong khoảng thời gian từ [Ngày bắt đầu] đến [Ngày kết thúc].
        - Dữ liệu đã được ẩn danh và tuân thủ các quy định về bảo mật thông tin bệnh nhân.
    """)

    st.subheader("Mô Tả Dữ Liệu")
    st.markdown("""
    - **(Thay thế phần này bằng mô tả chi tiết về các cột trong dữ liệu của bạn)**
    - Ví dụ:
        - **Age:** Tuổi của bệnh nhân (số nguyên).
        - **Lung Capacity:** Dung tích phổi của bệnh nhân (số thực, đơn vị: ...).
        - **Smoking Status_yes:** Tình trạng hút thuốc của bệnh nhân (1: Có, 0: Không).
        - **Hospital Visits:** Số lượt bệnh nhân đến khám tại bệnh viện (số nguyên).
        - **Disease Type_...:** Các cột này cho biết bệnh nhân có mắc loại bệnh phổi cụ thể nào không (1: Có, 0: Không).
        - **Recovered:** Bệnh nhân đã hồi phục chưa? (1: Có, 0: Chưa).
        - **Gender_female:** Giới tính bệnh nhân là nữ (1: Có, 0: Không/Khác).
        - **Gender_male**: Giới tính bệnh nhân là nam (1: Có, 0: Không/Khác).
    """)

    st.subheader("Dữ Liệu Thô (Mẫu)")
    st.dataframe(df.head()) # Show the first few rows

    with st.expander("Xem Toàn Bộ Dữ Liệu Thô"):
        st.dataframe(df)


# --- Page 2: Descriptive Statistics ---
elif page == "2. Thống Kê Mô Tả":
    st.header("2. Thống Kê Mô Tả")

    st.subheader("Thông Tin Dữ Liệu ")

    st.subheader("Phân Phối Dữ Liệu")
    numeric_cols = df.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        fig, ax = plt.subplots()
        sns.histplot(df[col].dropna(), kde=True, ax=ax)  # Drop NaN values before plotting
        st.pyplot(fig)
        with st.expander(f"Nhận xét về phân phối của {col}"):
          st.markdown(f"""
            - **(Thêm nhận xét về hình dạng phân phối, độ tập trung, độ phân tán, v.v. của {col})**
          """)

    st.subheader("Thống Kê Mô Tả (df.describe())")
    st.dataframe(df.describe())
    with st.expander("Nhận xét về Thống Kê Mô Tả"):
      st.markdown("""
          - **(Thêm nhận xét về các giá trị thống kê như trung bình, độ lệch chuẩn, min, max, các khoảng tứ phân vị, v.v.)**
      """)

# --- Page 3: In-Depth Analysis ---
elif page == "3. Phân Tích Chuyên Sâu":
    st.header("3. Phân Tích Chuyên Sâu")
    analysis_page = st.selectbox("Chọn Phân Tích", ["Thống kê chung", "Tuổi & Dung Tích Phổi", "Tỷ lệ phần trăm phục hồi theo loại bệnh", "Loại Bệnh", "Hút Thuốc & Dung Tích Phổi", "Lượt Khám Bệnh", "Tương Quan"])
    
    if analysis_page == "Thống kê chung":
        st.subheader("Thống Kê Chung")
        
            # We don't drop NaNs *here* because we want overall stats, including missingness
        total_patients = len(df)
        avg_age = df["Age"].mean()  # mean() automatically handles NaNs
        avg_lung_capacity = df["Lung Capacity"].mean()
        smoking_rate = df["Smoking Status_yes"].mean() * 100

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Tổng Số Bệnh Nhân", total_patients)
        with col2:
            st.metric("Tuổi Trung Bình", f"{round(avg_age, 2):.2f}" if not pd.isna(avg_age) else "N/A") # Handle potential NaN in average
        with col3:
            st.metric("Dung Tích Phổi Trung Bình", f"{round(avg_lung_capacity, 2):.2f}" if not pd.isna(avg_lung_capacity) else "N/A")
        with col4:
            st.metric("Tỷ Lệ Hút Thuốc (%)", f"{round(smoking_rate, 2):.2f}" if not pd.isna(smoking_rate) else "N/A")

        with st.expander("Xem Nhận Xét Chi Tiết"):
            st.markdown(f"""
            - **Tổng Số Bệnh Nhân:** {total_patients}
            - Cho biết quy mô của tập dữ liệu.

            - **Tuổi Trung Bình:** {f'{round(avg_age, 2):.2f}' if not pd.isna(avg_age) else 'N/A'}
            - Độ tuổi trung bình của bệnh nhân.

            - **Dung Tích Phổi Trung Bình:** {f'{round(avg_lung_capacity, 2):.2f}' if not pd.isna(avg_lung_capacity) else 'N/A'}
            - Dung tích phổi trung bình.

            - **Tỷ Lệ Hút Thuốc:** {f'{round(smoking_rate, 2):.2f}%' if not pd.isna(smoking_rate) else 'N/A'}
            - Phần trăm bệnh nhân hút thuốc.
            """)
        with st.expander("Xem Dữ Liệu"):
            st.dataframe(df)
        
    if analysis_page == "Tuổi & Dung Tích Phổi":
      st.subheader("Phân Bố Tuổi & Dung Tích Phổi")

      # Handle missing values BEFORE filtering (important!)
      filtered_df = df.dropna(subset=["Age", "Lung Capacity"])  # Remove rows where Age OR Lung Capacity is NaN

      # Age Range Slider (only if there are any non-null Ages)
      if not filtered_df["Age"].isnull().all():
        age_range = st.slider("Chọn Khoảng Tuổi", int(filtered_df["Age"].min()), int(filtered_df["Age"].max()), (int(filtered_df["Age"].min()), int(filtered_df["Age"].max())))
        filtered_df = filtered_df[(filtered_df["Age"] >= age_range[0]) & (filtered_df["Age"] <= age_range[1])]
      else:
        st.warning("Không có dữ liệu tuổi hợp lệ để hiển thị.")
        st.stop()

      fig, ax = plt.subplots(1, 2, figsize=(12, 5))
      sns.histplot(filtered_df["Age"], bins=20, kde=True, ax=ax[0])
      ax[0].set_title("Phân Bố Tuổi")
      ax[0].set_xlabel("Tuổi")
      ax[0].set_ylabel("Số Lượng")
      sns.histplot(filtered_df["Lung Capacity"], bins=20, kde=True, ax=ax[1])
      ax[1].set_title("Phân Bố Dung Tích Phổi")
      ax[1].set_xlabel("Dung Tích Phổi")
      ax[1].set_ylabel("Số Lượng")
      st.pyplot(fig)
      
      with st.expander("Xem Nhận Xét Chi Tiết"):
        st.markdown(f"""
        **Biểu đồ Phân Bố Tuổi:**
        - Biểu đồ này cho thấy sự phân bố độ tuổi của các bệnh nhân trong tập dữ liệu (đã lọc theo khoảng tuổi và loại bỏ các giá trị thiếu).
        - **Hình dạng phân bố:** Quan sát xem phân bố có đối xứng, lệch trái, lệch phải hay có nhiều đỉnh.  Ví dụ, nếu phân bố lệch phải, điều đó có nghĩa là có nhiều bệnh nhân lớn tuổi hơn.
        - **Độ tập trung:** Xem xét khoảng tuổi nào có tần suất xuất hiện cao nhất (đỉnh của biểu đồ).
        - **Đường KDE:** Đường cong KDE (Kernel Density Estimate) giúp ước lượng hàm mật độ xác suất của tuổi, cho thấy xu hướng chung của phân bố.

        **Biểu đồ Phân Bố Dung Tích Phổi:**
        - Biểu đồ này hiển thị sự phân bố dung tích phổi của các bệnh nhân (đã lọc theo khoảng tuổi và loại bỏ các giá trị thiếu).
        - **Hình dạng phân bố:** Tương tự như biểu đồ tuổi, quan sát hình dạng (đối xứng, lệch, nhiều đỉnh).
        - **Độ tập trung:** Xác định khoảng dung tích phổi phổ biến nhất.
        - **Độ phân tán:** Xem xét độ rộng của phân bố.  Phân bố rộng cho thấy sự khác biệt lớn về dung tích phổi giữa các bệnh nhân.  Phân bố hẹp cho thấy dung tích phổi tương đối đồng đều.
        - **Đường KDE:** Đường cong KDE ước tính mật độ xác suất, cho thấy xu hướng chung.
        - **So sánh với độ tuổi:** Có thể có mối liên hệ giữa độ tuổi và dung tích phổi.  Ví dụ, dung tích phổi có thể giảm dần theo tuổi.
        """)
      
    elif analysis_page == "Loại Bệnh":
        st.subheader("Phân Bố Loại Bệnh")
        disease_counts = df[disease_cols].sum()
        st.bar_chart(disease_counts)
        
        with st.expander("Xem Nhận Xét Chi Tiết"):
            st.markdown("""
            **Biểu đồ Cột Phân Bố Loại Bệnh:**
            - Biểu đồ này thể hiện số lượng bệnh nhân mắc mỗi loại bệnh phổi được ghi nhận trong dữ liệu.
            - **So sánh tần suất:** Xác định loại bệnh nào phổ biến nhất (cột cao nhất) và loại bệnh nào ít gặp nhất (cột thấp nhất).
            - **Chênh lệch:** Đánh giá mức độ chênh lệch về số lượng bệnh nhân giữa các loại bệnh. Sự chênh lệch lớn có thể gợi ý về các yếu tố nguy cơ hoặc đặc điểm dịch tễ của từng bệnh.
            - **Tổng số bệnh nhân:** Lưu ý rằng tổng số lượng bệnh nhân trên biểu đồ này có thể lớn hơn tổng số bệnh nhân trong tập dữ liệu, vì một bệnh nhân có thể mắc nhiều loại bệnh cùng lúc.
            """)
        
    elif analysis_page == "Hút Thuốc & Dung Tích Phổi":
      st.subheader("Ảnh Hưởng của Hút Thuốc lên Dung Tích Phổi")

      # --- Handle missing values BEFORE plotting ---
      plot_df = df.dropna(subset=["Smoking Status_yes", "Lung Capacity"])

      # Boxplot
      fig1, ax1 = plt.subplots(figsize=(6, 4))
      sns.boxplot(x=plot_df["Smoking Status_yes"], y=plot_df["Lung Capacity"], ax=ax1)
      ax1.set_xticklabels(["Không Hút Thuốc", "Hút Thuốc"])
      ax1.set_xlabel("Tình Trạng Hút Thuốc")
      ax1.set_ylabel("Dung Tích Phổi")
      st.pyplot(fig1)
      
      with st.expander("Xem Nhận Xét Chi Tiết (Biểu đồ Hộp)"):
        st.markdown("""
        **Biểu đồ Hộp (Boxplot):**
        - Biểu đồ này so sánh phân bố dung tích phổi giữa hai nhóm: người không hút thuốc và người hút thuốc (sau khi loại bỏ các giá trị thiếu).
        - **Trung vị:** So sánh vị trí đường kẻ giữa (trung vị) của hai hộp. Nếu trung vị của nhóm hút thuốc thấp hơn, điều đó cho thấy dung tích phổi trung bình của nhóm này thấp hơn.
        - **IQR (Khoảng tứ phân vị):** So sánh độ cao của hai hộp. IQR lớn hơn cho thấy sự phân tán dữ liệu lớn hơn (dung tích phổi có nhiều biến động hơn).
        - **Râu:** So sánh độ dài của râu. Râu dài hơn cho thấy phạm vi dung tích phổi rộng hơn.
        - **Điểm ngoại lai:** Các điểm nằm ngoài râu là các giá trị bất thường. Xem xét liệu có nhiều điểm ngoại lai ở một trong hai nhóm hay không.
        - **Kết luận sơ bộ:** Dựa trên so sánh trực quan, có thể đưa ra kết luận sơ bộ về ảnh hưởng của hút thuốc lên dung tích phổi.
        """)
        
      # Scatter Plot
      fig2, ax2 = plt.subplots(figsize=(8, 6))
      sns.scatterplot(x=plot_df["Smoking Status_yes"], y=plot_df["Lung Capacity"], data=plot_df, ax=ax2, alpha=0.3)
      ax2.set_xlabel("Tình Trạng Hút Thuốc (0: Không Hút, 1: Có Hút)")
      ax2.set_ylabel("Dung Tích Phổi")
      st.pyplot(fig2)
      
      with st.expander("Xem Nhận Xét Chi Tiết (Biểu đồ Phân Tán)"):
        st.markdown("""
        **Biểu đồ Phân Tán (Scatter Plot):**
        - Biểu đồ này thể hiện mối quan hệ giữa tình trạng hút thuốc (0: Không hút, 1: Có hút) và dung tích phổi của từng bệnh nhân (sau khi loại bỏ giá trị thiếu).
        - **Mỗi điểm:** Mỗi điểm trên biểu đồ đại diện cho một bệnh nhân.
        - **Xu hướng:** Quan sát xem có xu hướng nào không (ví dụ: các điểm có xu hướng dốc xuống khi x tăng không).
        - **Phân tán:**  Độ phân tán của các điểm cho biết mức độ mạnh yếu của mối quan hệ.
        - **Kết hợp với Boxplot:**  Kết hợp thông tin từ biểu đồ phân tán và biểu đồ hộp để có cái nhìn toàn diện hơn.
        """)

      # T-test (fully manual, no scipy)
      smokers = plot_df[plot_df["Smoking Status_yes"] == 1]["Lung Capacity"]
      non_smokers = plot_df[plot_df["Smoking Status_yes"] == 0]["Lung Capacity"]

      if len(smokers) > 0 and len(non_smokers) > 0:
          # Calculate t-statistic and p-value manually
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
                - **p-value:** {p_value:.3f} (Xác suất của kết quả nếu không có sự khác biệt thực sự).
                - **Ý nghĩa thống kê:** p < 0.05: Có sự khác biệt. p >= 0.05: Không đủ bằng chứng.
                """)
      else:
          st.write("Không thể thực hiện kiểm định t-test: Một hoặc cả hai nhóm không có dữ liệu.")
        
    elif analysis_page == "Lượt Khám Bệnh":
      st.subheader("Lượt Khám Bệnh Trung Bình theo Loại Bệnh")

      # Multiselect for disease types
      selected_diseases = st.multiselect("Chọn Loại Bệnh", disease_cols, default=disease_cols)

      if selected_diseases:
          # Handle missing values *before* calculating the mean
            visits_per_disease = {col: df[df[col] == 1]["Hospital Visits"].dropna().mean() for col in selected_diseases}
            st.bar_chart(pd.Series(visits_per_disease))
            with st.expander("Xem Nhận Xét Chi Tiết"):
                st.markdown("""
                    **Biểu đồ Cột Lượt Khám Bệnh Trung Bình:**
                    - Biểu đồ này hiển thị số lượt khám bệnh trung bình cho các loại bệnh phổi *đã chọn*.
                    - **So sánh:** So sánh chiều cao của các cột.
                    - **Giá trị cụ thể:** Chú ý đến các giá trị trung bình.
                    - **Lưu ý:** Số lượt khám trung bình cao có thể do nhiều nguyên nhân.
                    - **Kết hợp:** Nên kết hợp thông tin với các biểu đồ khác.
                """)
      else:
          st.write("Vui lòng chọn ít nhất một loại bệnh.")
        
    elif analysis_page == "Tương Quan":
        st.subheader("Biểu Đồ Tương Quan (Heatmap)")

        # Calculate correlation matrix, handling missing values appropriately.
        corr_matrix = df.corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        st.pyplot(fig)
        with st.expander("Xem Nhận Xét Chi Tiết"):
            st.markdown("""
            **Biểu đồ Nhiệt Tương Quan (Heatmap):**
            - Biểu đồ này thể hiện *mối tương quan tuyến tính* giữa các cặp biến số.
            - **Giải thích:** Gần 1: Tương quan dương mạnh. Gần -1: Tương quan âm mạnh. Gần 0: Yếu.
            - **Màu sắc:** Đỏ: Dương. Xanh: Âm. Đậm: Mạnh.
            - **Cảnh báo:** Tương quan không phải nhân quả. Chỉ tuyến tính.
            - **Cách sử dụng:** Tìm cặp biến tương quan mạnh.
            """)
    

# --- Page 4: Overall Conclusions ---
elif page == "4. Nhận Xét Chung":
    avg_age = df["Age"].mean()
    smoking_rate = df["Smoking Status_yes"].mean() * 100
    avg_hospital_visits = df["Hospital Visits"].mean()
    male_percentage = df["Gender_male"].mean() * 100
    female_percentage = df["Gender_female"].mean() * 100
    
    st.header("4. Nhận Xét Chung")

    st.markdown(f"""
    - **Tổng Quan về Dữ Liệu và Kết Quả Phân Tích:**
        - **Mối Tương Quan giữa Hút Thuốc và Dung Tích Phổi:**
            - Dữ liệu cho thấy mối tương quan đáng kể giữa hút thuốc và dung tích phổi. Cụ thể, những **bệnh nhân hút thuốc** có xu hướng có **dung tích phổi thấp hơn** so với những bệnh nhân không hút thuốc.
            - Điều này có thể được giải thích bởi tác động tiêu cực của khói thuốc lá lên hệ hô hấp, gây ra các bệnh lý như viêm phế quản mãn tính và COPD.
        - **Phổ Biến của Bệnh Bronchitis:**
            - Bệnh **Bronchitis (Viêm phế quản)** là loại bệnh phổi **phổ biến nhất** trong tập dữ liệu. Điều này có thể phản ánh tình trạng ô nhiễm không khí hoặc thói quen hút thuốc lá phổ biến trong cộng đồng.
        - **Tuổi Trung Bình của Bệnh Nhân:**
            - Tuổi trung bình của bệnh nhân là **{avg_age:.2f} tuổi**, và có sự khác biệt về dung tích phổi giữa các nhóm tuổi. Nhóm tuổi lớn hơn có xu hướng có dung tích phổi thấp hơn, có thể do sự suy giảm chức năng phổi theo tuổi tác.
        - **Tình Trạng Hút Thuốc:**
            - Tỷ lệ bệnh nhân hút thuốc trong tập dữ liệu là **{smoking_rate:.2f}%**. Điều này có thể cho thấy mức độ phổ biến của thói quen hút thuốc trong cộng đồng bệnh nhân.
        - **Số Lượt Khám Bệnh:**
            - Số lượt khám bệnh trung bình của bệnh nhân là **{avg_hospital_visits:.2f}**. Điều này có thể phản ánh mức độ nghiêm trọng của bệnh và nhu cầu chăm sóc y tế của bệnh nhân.
        - **Giới Tính:**
            - Tỷ lệ bệnh nhân nam và nữ trong tập dữ liệu là **{male_percentage:.2f}%** và **{female_percentage:.2f}%**. Điều này có thể cho thấy sự phân bố giới tính trong cộng đồng bệnh nhân.
            Trong những năm gần đây, tỷ lệ hút thuốc ở nữ giới có xu hướng tăng lên, điều này có thể dẫn đến sự gia tăng các bệnh lý liên quan đến phổi ở nữ giới. Ngoài ra các nghiên cứu cũng
            chỉ ra rằng nữ giới có thể nhạy cảm hơn với các tác động tiêu cực của khói thuốc lá so với nam giới, dẫn đến nguy cơ cao hơn mắc các bệnh phổi.
        - **Nhu Cầu Nghiên Cứu Thêm:**
            - Cần có thêm nghiên cứu để xác định nguyên nhân và các yếu tố nguy cơ của các bệnh phổi này. Điều này bao gồm việc xem xét các yếu tố môi trường, di truyền và lối sống khác có thể ảnh hưởng đến sức khỏe phổi.
    """)
    
    st.subheader("Hạn Chế")
    st.markdown("""
      - **Hạn Chế của Dữ Liệu và Phân Tích:**
        - **Kích Thước Mẫu:**
            - Kích thước mẫu có thể không đủ lớn để đưa ra kết luận chắc chắn. Một mẫu lớn hơn sẽ cung cấp kết quả đáng tin cậy hơn và giảm thiểu sai số thống kê.
        - **Thiếu Sót Dữ Liệu:**
            - Dữ liệu có thể bị thiếu sót hoặc không đầy đủ, ảnh hưởng đến độ chính xác của phân tích. Ví dụ, một số bệnh nhân có thể không cung cấp đầy đủ thông tin về tình trạng hút thuốc hoặc dung tích phổi.
        - **Biến Quan Trọng Chưa Được Thu Thập:**
            - Một số biến quan trọng có thể chưa được thu thập hoặc xem xét trong phân tích. Các yếu tố như chế độ ăn uống, môi trường sống, và tiền sử bệnh lý gia đình có thể ảnh hưởng đến sức khỏe phổi nhưng không có trong dữ liệu.
        - **Tương Quan Không Đồng Nghĩa với Nhân Quả:**
            - Tương quan không đồng nghĩa với nhân quả; cần có thêm nghiên cứu để xác định mối quan hệ nhân quả. Ví dụ, mặc dù có mối tương quan giữa hút thuốc và dung tích phổi, không thể kết luận rằng hút thuốc là nguyên nhân duy nhất gây ra giảm dung tích phổi.
    """)

    st.subheader("Đề Xuất Nghiên Cứu Tiếp Theo")
    st.markdown("""
      - **Đề Xuất cho Nghiên Cứu Tiếp Theo:**
        - **Thu Thập Thêm Dữ Liệu:**
            - Thu thập thêm dữ liệu để tăng kích thước mẫu và độ tin cậy của kết quả. Điều này bao gồm việc mở rộng phạm vi nghiên cứu để bao gồm nhiều bệnh nhân hơn từ các khu vực địa lý khác nhau.
        - **Bổ Sung Các Biến Quan Trọng:**
            - Bổ sung các biến quan trọng khác có thể ảnh hưởng đến bệnh phổi, như chế độ ăn uống, môi trường sống, v.v. Điều này sẽ giúp hiểu rõ hơn về các yếu tố nguy cơ và bảo vệ sức khỏe phổi.
        - **Nghiên Cứu Sâu Hơn về Mối Quan Hệ Nhân Quả:**
            - Thực hiện các nghiên cứu sâu hơn để xác định mối quan hệ nhân quả giữa các yếu tố nguy cơ và bệnh phổi. Sử dụng các phương pháp nghiên cứu như thử nghiệm lâm sàng hoặc nghiên cứu theo dõi dài hạn để xác định nguyên nhân chính xác.
        - **Sử Dụng Các Phương Pháp Phân Tích Tiên Tiến:**
            - Sử dụng các phương pháp phân tích tiên tiến hơn để khám phá các mối quan hệ phức tạp trong dữ liệu. Các phương pháp như phân tích đa biến, mô hình học máy, và phân tích mạng có thể giúp phát hiện các mối quan hệ ẩn và dự đoán nguy cơ bệnh phổi.
    """)