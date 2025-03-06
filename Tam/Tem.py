import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set color palette for colorblind-friendly visuals
sns.set_palette("colorblind")

# Load data
file_path = "lung_disease_data_preprocessed.csv"

try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    st.error(f"Lỗi: Không tìm thấy tệp dữ liệu tại {file_path}. Vui lòng kiểm tra lại đường dẫn.")
    st.stop()
except Exception as e:
    st.error(f"Đã xảy ra lỗi không mong muốn: {e}")
    st.stop()

# Extract disease columns
disease_cols = [col for col in df.columns if "Disease Type" in col]

# Streamlit App
st.title("🫁 Bảng Điều Khiển Bệnh Phổi")

# Initialize session state
if "page" not in st.session_state:
    st.session_state["page"] = "📊 Tuổi & Dung Tích Phổi"

# --- Navigation (Selectbox) ---
page_options = ["📈 Thống Kê Chung", "📊 Tuổi & Dung Tích Phổi", "🦠 Loại Bệnh", "🚬 Hút Thuốc & Dung Tích Phổi", "🏥 Lượt Khám Bệnh", "🔍 Phân Tích Tương Quan"]
selected_page = st.selectbox("Chọn Trang", page_options)
st.session_state["page"] = selected_page


# --- Content ---
if st.session_state["page"] == "📊 Tuổi & Dung Tích Phổi":
    st.subheader("📊 Phân Bố Tuổi & Dung Tích Phổi")

    # Age Range Slider
    age_range = st.slider("Chọn Khoảng Tuổi", int(df["Age"].min()), int(df["Age"].max()), (int(df["Age"].min()), int(df["Age"].max())))
    filtered_df = df[(df["Age"] >= age_range[0]) & (df["Age"] <= age_range[1])]

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
        - Biểu đồ này cho thấy sự phân bố độ tuổi của các bệnh nhân trong tập dữ liệu (đã lọc theo khoảng tuổi {age_range[0]} - {age_range[1]}).
        - **Hình dạng phân bố:** Quan sát xem phân bố có đối xứng, lệch trái, lệch phải hay có nhiều đỉnh.  Ví dụ, nếu phân bố lệch phải, điều đó có nghĩa là có nhiều bệnh nhân lớn tuổi hơn.
        - **Độ tập trung:** Xem xét khoảng tuổi nào có tần suất xuất hiện cao nhất (đỉnh của biểu đồ).
        - **Đường KDE:** Đường cong KDE (Kernel Density Estimate) giúp ước lượng hàm mật độ xác suất của tuổi, cho thấy xu hướng chung của phân bố.

        **Biểu đồ Phân Bố Dung Tích Phổi:**
        - Biểu đồ này hiển thị sự phân bố dung tích phổi của các bệnh nhân (đã lọc theo khoảng tuổi {age_range[0]} - {age_range[1]}).
        - **Hình dạng phân bố:** Tương tự như biểu đồ tuổi, quan sát hình dạng (đối xứng, lệch, nhiều đỉnh).
        - **Độ tập trung:** Xác định khoảng dung tích phổi phổ biến nhất.
        - **Độ phân tán:** Xem xét độ rộng của phân bố.  Phân bố rộng cho thấy sự khác biệt lớn về dung tích phổi giữa các bệnh nhân.  Phân bố hẹp cho thấy dung tích phổi tương đối đồng đều.
        - **Đường KDE:** Đường cong KDE ước tính mật độ xác suất, cho thấy xu hướng chung.
        - **So sánh với độ tuổi:** Có thể có mối liên hệ giữa độ tuổi và dung tích phổi.  Ví dụ, dung tích phổi có thể giảm dần theo tuổi.
        """)

elif st.session_state["page"] == "🦠 Loại Bệnh":
    st.subheader("🦠 Phân Bố Loại Bệnh")
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

elif st.session_state["page"] == "🚬 Hút Thuốc & Dung Tích Phổi":
    st.subheader("🚬 Ảnh Hưởng của Hút Thuốc lên Dung Tích Phổi")

    # Boxplot
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    sns.boxplot(x=df["Smoking Status_yes"], y=df["Lung Capacity"], ax=ax1)
    ax1.set_xticklabels(["Không Hút Thuốc", "Hút Thuốc"])
    ax1.set_xlabel("Tình Trạng Hút Thuốc")
    ax1.set_ylabel("Dung Tích Phổi")
    st.pyplot(fig1)

    with st.expander("Xem Nhận Xét Chi Tiết (Biểu đồ Hộp)"):
        st.markdown("""
        **Biểu đồ Hộp (Boxplot):**
        - Biểu đồ này so sánh phân bố dung tích phổi giữa hai nhóm: người không hút thuốc và người hút thuốc.
        - **Trung vị:** So sánh vị trí đường kẻ giữa (trung vị) của hai hộp. Nếu trung vị của nhóm hút thuốc thấp hơn, điều đó cho thấy dung tích phổi trung bình của nhóm này thấp hơn.
        - **IQR (Khoảng tứ phân vị):** So sánh độ cao của hai hộp. IQR lớn hơn cho thấy sự phân tán dữ liệu lớn hơn (dung tích phổi có nhiều biến động hơn).
        - **Râu:** So sánh độ dài của râu. Râu dài hơn cho thấy phạm vi dung tích phổi rộng hơn.
        - **Điểm ngoại lai:** Các điểm nằm ngoài râu là các giá trị bất thường. Xem xét liệu có nhiều điểm ngoại lai ở một trong hai nhóm hay không.
        - **Kết luận sơ bộ:** Dựa trên so sánh trực quan, có thể đưa ra kết luận sơ bộ về ảnh hưởng của hút thuốc lên dung tích phổi.
        """)

    # Scatter Plot
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x="Smoking Status_yes", y="Lung Capacity", data=df, ax=ax2, alpha=0.3)
    ax2.set_xlabel("Tình Trạng Hút Thuốc (0: Không Hút, 1: Có Hút)")
    ax2.set_ylabel("Dung Tích Phổi")
    st.pyplot(fig2)

    with st.expander("Xem Nhận Xét Chi Tiết (Biểu đồ Phân Tán)"):
        st.markdown("""
        **Biểu đồ Phân Tán (Scatter Plot):**
        - Biểu đồ này thể hiện mối quan hệ giữa tình trạng hút thuốc (0: Không hút, 1: Có hút) và dung tích phổi của từng bệnh nhân.
        - **Mỗi điểm:** Mỗi điểm trên biểu đồ đại diện cho một bệnh nhân.
        - **Xu hướng:** Quan sát xem có xu hướng nào không (ví dụ: các điểm có xu hướng dốc xuống khi x tăng không).
        - **Phân tán:**  Độ phân tán của các điểm cho biết mức độ mạnh yếu của mối quan hệ.
        - **Kết hợp với Boxplot:**  Kết hợp thông tin từ biểu đồ phân tán và biểu đồ hộp để có cái nhìn toàn diện hơn.
        """)

    # T-test (fully manual, no scipy)
    smokers = df[df["Smoking Status_yes"] == 1]["Lung Capacity"]
    non_smokers = df[df["Smoking Status_yes"] == 0]["Lung Capacity"]

    if len(smokers) > 0 and len(non_smokers) > 0:
        # Calculate t-statistic and p-value manually
        mean_smokers = np.mean(smokers)
        mean_non_smokers = np.mean(non_smokers)
        std_smokers = np.std(smokers, ddof=1)  # Sample standard deviation
        std_non_smokers = np.std(non_smokers, ddof=1)
        n_smokers = len(smokers)
        n_non_smokers = len(non_smokers)

        # Pooled standard deviation
        sp = np.sqrt(((n_smokers - 1) * std_smokers**2 + (n_non_smokers - 1) * std_non_smokers**2) / (n_smokers + n_non_smokers - 2))

        # t-statistic
        t_stat = (mean_smokers - mean_non_smokers) / (sp * np.sqrt(1/n_smokers + 1/n_non_smokers))

        # Degrees of freedom
        df_ttest = n_smokers + n_non_smokers - 2

        # Approximate p-value (two-tailed) using the standard normal CDF
        #  We'll use an approximation based on the Z-distribution (standard normal)
        #  This is valid for larger sample sizes due to the Central Limit Theorem.
        def z_to_p(z):
            """Approximates the two-tailed p-value from a Z-score."""
            # Use a lookup table (or an approximation function) for the standard normal CDF.
            # For simplicity, we use a simplified approximation.  A more accurate
            # approach would use a more precise approximation or a lookup table.
            z = abs(z)  # Ensure z is positive
            if z > 3.7:  # Beyond this, p is very small
                return 0.0
            # Very simplified approximation (good enough for demonstration)
            p = 1 / (1 + np.exp(0.07056 * z**3 + 1.5976 * z))
            return 2 * (1 - p)  # Two-tailed p-value

        p_value = z_to_p(t_stat)


        st.write(f"Kiểm định t-test: t-statistic = {t_stat:.2f}, p-value = {p_value:.3f}")
        with st.expander("Giải thích kết quả t-test"):
            st.markdown(f"""
              - **t-statistic:** {t_stat:.2f} (Đo lường sự khác biệt giữa trung bình của hai nhóm, tính theo đơn vị sai số chuẩn). Giá trị t càng lớn (dương hoặc âm) thì sự khác biệt càng lớn.
              - **p-value:** {p_value:.3f} (Xác suất quan sát được sự khác biệt lớn như vậy (hoặc lớn hơn) giữa hai nhóm, nếu giả thuyết không (không có sự khác biệt thực sự) là đúng).
              - **Ý nghĩa thống kê:**
                - Nếu p-value < 0.05 (ngưỡng ý nghĩa thường dùng): Có bằng chứng thống kê để bác bỏ giả thuyết không (null hypothesis). Kết luận: Có sự khác biệt có *ý nghĩa thống kê* về dung tích phổi giữa người hút thuốc và không hút thuốc. Điều này *gợi ý* rằng hút thuốc có thể ảnh hưởng đến dung tích phổi.
                - Nếu p-value >= 0.05: Không có đủ bằng chứng thống kê để bác bỏ giả thuyết không. Kết luận: Không thể kết luận có sự khác biệt có ý nghĩa thống kê về dung tích phổi giữa hai nhóm dựa trên dữ liệu này. *Lưu ý: Điều này không có nghĩa là không có sự khác biệt, chỉ là không đủ bằng chứng từ dữ liệu.*
              - **Quan trọng:** Kiểm định t-test chỉ cho biết *có* sự khác biệt có ý nghĩa thống kê hay không, chứ không khẳng định nguyên nhân và kết quả.
              """)
    else:
        st.write("Không thể thực hiện kiểm định t-test: Một hoặc cả hai nhóm (người hút thuốc/không hút thuốc) không có dữ liệu.")

elif st.session_state["page"] == "🏥 Lượt Khám Bệnh":
    st.subheader("🏥 Lượt Khám Bệnh Trung Bình theo Loại Bệnh")
    # Multiselect for disease types
    selected_diseases = st.multiselect("Chọn Loại Bệnh", disease_cols, default=disease_cols)

    if selected_diseases:
        visits_per_disease = {col: df[df[col] == 1]["Hospital Visits"].mean() for col in selected_diseases}
        st.bar_chart(pd.Series(visits_per_disease))
        with st.expander("Xem Nhận Xét Chi Tiết"):
            st.markdown("""
                **Biểu đồ Cột Lượt Khám Bệnh Trung Bình:**
                - Biểu đồ này hiển thị số lượt khám bệnh trung bình cho các loại bệnh phổi *đã chọn*.
                - **So sánh:** So sánh chiều cao của các cột để xem loại bệnh nào có số lượt khám trung bình cao hơn (gợi ý về mức độ nghiêm trọng hoặc tần suất tái phát).
                - **Giá trị cụ thể:** Chú ý đến các giá trị trung bình (có thể di chuột qua các cột để xem giá trị chính xác).
                - **Lưu ý:** Số lượt khám trung bình cao có thể do nhiều nguyên nhân:
                    - Bệnh nặng hơn, cần điều trị tích cực hơn.
                    - Bệnh có xu hướng tái phát thường xuyên.
                    - Bệnh nhân cần theo dõi định kỳ.
                    - Các yếu tố khác không được thể hiện trong dữ liệu (ví dụ: độ tuổi, bệnh nền).
                - **Kết hợp với thông tin khác:** Nên kết hợp thông tin từ biểu đồ này với các biểu đồ khác (ví dụ: phân bố loại bệnh, tương quan) để có cái nhìn toàn diện hơn.
            """)
    else:
        st.write("Vui lòng chọn ít nhất một loại bệnh.")

elif st.session_state["page"] == "📈 Thống Kê Chung":
    st.subheader("📈 Thống Kê Chung")
    total_patients = len(df)
    avg_age = df["Age"].mean()
    avg_lung_capacity = df["Lung Capacity"].mean()
    smoking_rate = df["Smoking Status_yes"].mean() * 100

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Tổng Số Bệnh Nhân", total_patients)
    with col2:
        st.metric("Tuổi Trung Bình", round(avg_age, 2))
    with col3:
        st.metric("Dung Tích Phổi Trung Bình", round(avg_lung_capacity, 2))
    with col4:
        st.metric("Tỷ Lệ Hút Thuốc (%)", round(smoking_rate, 2))

    with st.expander("Xem Nhận Xét Chi Tiết"):
        st.markdown(f"""
        - **Tổng Số Bệnh Nhân:** {total_patients}
          - Cho biết quy mô của tập dữ liệu. Số lượng bệnh nhân càng lớn, kết quả phân tích càng có độ tin cậy cao (nếu dữ liệu được thu thập một cách ngẫu nhiên và đại diện).

        - **Tuổi Trung Bình:** {round(avg_age, 2)}
          - Cung cấp thông tin về độ tuổi trung bình của các bệnh nhân. Giá trị này có thể được so sánh với độ tuổi trung bình của dân số nói chung (nếu có) để đánh giá xem nhóm bệnh nhân trong tập dữ liệu có độ tuổi cao hơn, thấp hơn hay tương đương.

        - **Dung Tích Phổi Trung Bình:** {round(avg_lung_capacity, 2)}
          - Cho biết giá trị trung bình của dung tích phổi trong tập dữ liệu. Giá trị này có thể được so sánh với các giá trị tham chiếu từ các nghiên cứu khác (nếu có) hoặc với các nhóm bệnh nhân khác để đánh giá mức độ suy giảm chức năng phổi.

        - **Tỷ Lệ Hút Thuốc:** {round(smoking_rate, 2)}%
          - Thể hiện phần trăm bệnh nhân có tiền sử hút thuốc. Tỷ lệ này có thể được so sánh với tỷ lệ hút thuốc trong dân số nói chung để xem liệu nhóm bệnh nhân này có tỷ lệ hút thuốc cao hơn hay không.  Tỷ lệ cao hơn có thể *gợi ý* (nhưng không khẳng định) mối liên hệ giữa hút thuốc và bệnh phổi.
        """)
    with st.expander("Xem Dữ Liệu"):
        st.dataframe(df)

elif st.session_state["page"] == "🔍 Phân Tích Tương Quan":
    st.subheader("🔍 Biểu Đồ Tương Quan (Heatmap)")
    corr_matrix = df.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)
    with st.expander("Xem Nhận Xét Chi Tiết"):
        st.markdown("""
          **Biểu đồ Nhiệt Tương Quan (Heatmap):**
          - Biểu đồ này thể hiện *mối tương quan tuyến tính* giữa tất cả các cặp biến số (numeric variables) trong tập dữ liệu.
          - **Giải thích các giá trị:**
            - Giá trị gần 1: Tương quan dương mạnh (khi một biến tăng thì biến kia *có xu hướng* tăng theo).
            - Giá trị gần -1: Tương quan âm mạnh (khi một biến tăng thì biến kia *có xu hướng* giảm).
            - Giá trị gần 0: Tương quan tuyến tính yếu hoặc không có tương quan tuyến tính.
          - **Màu sắc:**
            - Màu đỏ: Tương quan dương.  Càng đậm, tương quan càng mạnh.
            - Màu xanh: Tương quan âm. Càng đậm, tương quan càng mạnh.
            - Màu trắng/nhạt: Tương quan yếu.
          - **Cảnh báo:**
            - *Tương quan không phải là quan hệ nhân quả.* Hai biến có tương quan mạnh không có nghĩa là biến này gây ra biến kia. Có thể có một biến thứ ba (biến gây nhiễu) ảnh hưởng đến cả hai.
            - *Chỉ thể hiện tương quan tuyến tính.* Có thể có các mối quan hệ phi tuyến (ví dụ: hình chữ U) mà heatmap không thể hiện được.
          - **Cách sử dụng:**
            - Tìm các cặp biến có tương quan mạnh (gần 1 hoặc -1) để xem xét kỹ hơn.  Ví dụ:
              - Có thể có tương quan âm mạnh giữa `Smoking Status_yes` và `Lung Capacity`.
              - Có thể có tương quan dương giữa `Age` và `Hospital Visits`.
            - Xem xét các biến có tương quan yếu (gần 0) để loại bỏ khỏi các phân tích sâu hơn (nếu mục tiêu là tìm các yếu tố *ảnh hưởng mạnh*).
          """)

# Check for missing data (optional, but good practice)
if df.isnull().sum().sum() > 0:
    st.warning(f"Số lượng giá trị thiếu: {df.isnull().sum().sum()}")
    st.write("Các dòng có giá trị thiếu:")
    st.dataframe(df[df.isnull().any(axis=1)])