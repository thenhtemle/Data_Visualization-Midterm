import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Định nghĩa bảng màu dành cho người mù màu đỏ
color_palette = ["#88CCEE", "#44AA99", "#117733", "#999933", "#DDCC77", "#CC6677", "#882255", "#AA4499"]

@st.cache_data
def load_data():
    return pd.read_csv("lung_disease_data_cleaned.csv")

data = load_data()

# Streamlit dashboard
st.title("📊 Phân Tích Bệnh Phổi 🫁")

# Sidebar filters
st.sidebar.header("Bộ Lọc")
age_range = st.sidebar.slider("Chọn độ tuổi", int(data["Tuổi"].min()), int(data["Tuổi"].max()), (30, 70))
gender = st.sidebar.multiselect("Chọn giới tính", options=data["Giới Tính"].unique(), default=data["Giới Tính"].unique())
smoking_status = st.sidebar.multiselect("Tình trạng hút thuốc", options=data["Tình Trạng Hút Thuốc"].unique(), default=data["Tình Trạng Hút Thuốc"].unique())
disease_type = st.sidebar.multiselect("Loại bệnh", options=data["Loại Bệnh"].unique(), default=data["Loại Bệnh"].unique())

filtered_data = data[(data["Tuổi"] >= age_range[0]) & (data["Tuổi"] <= age_range[1]) &
                      (data["Giới Tính"].isin(gender)) &
                      (data["Tình Trạng Hút Thuốc"].isin(smoking_status)) &
                      (data["Loại Bệnh"].isin(disease_type))]

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Phân bố tuổi theo loại bệnh",
    "🚬 Mối quan hệ hút thuốc & dung lượng phổi",
    "🏥 Số lần khám theo loại điều trị",
    "🔁 Tỷ lệ phục hồi theo loại bệnh",
    "📊 Phân bố giới tính trong từng bệnh"
])

# Biểu đồ 1
with tab1:
    st.subheader("Phân bố tuổi theo loại bệnh")
    if not filtered_data.empty:
        fig, ax = plt.subplots()
        sns.boxplot(x="Loại Bệnh", y="Tuổi", data=filtered_data, ax=ax, palette=color_palette)
        plt.xticks(rotation=45)
        st.pyplot(fig)
        avg_age = filtered_data.groupby("Loại Bệnh")["Tuổi"].mean()
        if not avg_age.empty:
            highest_disease = avg_age.idxmax()
            lowest_disease = avg_age.idxmin()
            st.markdown(f"📌 **Nhận xét:** Người mắc **{highest_disease}** có độ tuổi trung bình cao nhất (**{avg_age.max():.2f}** tuổi), trong khi người mắc **{lowest_disease}** có độ tuổi trung bình thấp nhất (**{avg_age.min():.2f}** tuổi).")

# Biểu đồ 2
with tab2:
    st.subheader("Mối quan hệ giữa hút thuốc và dung lượng phổi")
    if not filtered_data.empty and "Dung Lượng Phổi" in filtered_data.columns:
        fig, ax = plt.subplots()
        sns.boxplot(x="Tình Trạng Hút Thuốc", y="Dung Lượng Phổi", data=filtered_data, ax=ax, palette=color_palette)
        st.pyplot(fig)
        mean_lung_capacity = filtered_data.groupby("Tình Trạng Hút Thuốc")["Dung Lượng Phổi"].mean()
        if not mean_lung_capacity.empty:
            if set(["Có", "Không"]).issubset(mean_lung_capacity.index):
                diff = mean_lung_capacity["Không"] - mean_lung_capacity["Có"]
                st.markdown(f"🔍 **Nhận xét:** Người không hút thuốc có dung lượng phổi trung bình cao hơn người hút thuốc khoảng **{diff:.2f}** lít." if diff > 0 else "🔍 **Nhận xét:** Người hút thuốc có dung lượng phổi trung bình tương đương hoặc cao hơn người không hút thuốc.")

# Biểu đồ 3
with tab3:
    st.subheader("Số lần khám trung bình theo loại điều trị")
    if "Số Lần Khám" in filtered_data.columns:
        avg_visits = filtered_data.groupby("Loại Điều Trị")["Số Lần Khám"].mean().reset_index()
        if not avg_visits.empty:
            st.bar_chart(avg_visits.set_index("Loại Điều Trị"))
            max_treatment = avg_visits.loc[avg_visits["Số Lần Khám"].idxmax()]
            min_treatment = avg_visits.loc[avg_visits["Số Lần Khám"].idxmin()]
            st.markdown(f"📌 **Nhận xét:** Loại điều trị có số lần khám trung bình cao nhất là **{max_treatment['Loại Điều Trị']}** với **{max_treatment['Số Lần Khám']:.2f}** lần. Ngược lại, **{min_treatment['Loại Điều Trị']}** có số lần khám thấp nhất với **{min_treatment['Số Lần Khám']:.2f}** lần.")

# Biểu đồ 4
with tab4:
    st.subheader("Tỷ lệ phục hồi theo loại bệnh")
    if "Phục Hồi" in filtered_data.columns:
        recovery_rate = filtered_data.groupby("Loại Bệnh")["Phục Hồi"].value_counts(normalize=True).unstack()
        if not recovery_rate.empty:
            st.bar_chart(recovery_rate)
            st.markdown("🔍 **Nhận xét:** Dữ liệu phục hồi được hiển thị theo loại bệnh.")

# Biểu đồ 5
with tab5:
    st.subheader("Phân bố giới tính trong từng bệnh")
    if not filtered_data.empty:
        fig, ax = plt.subplots()
        sns.countplot(x="Loại Bệnh", hue="Giới Tính", data=filtered_data, ax=ax, palette=color_palette)
        plt.xticks(rotation=45)
        st.pyplot(fig)

st.write("Dữ liệu đã được lọc theo các tiêu chí bạn chọn. Hãy thay đổi bộ lọc để xem xu hướng khác nhau! 📊")
