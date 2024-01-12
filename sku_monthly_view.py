import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
import numpy as np
from streamlit_tags import st_tags

st.set_page_config(layout="wide",page_title='SPU数据跟踪',page_icon = 'favicon.png')
conn = st.connection("gsheets", type=GSheetsConnection)

# 加载谷歌表格数据
@st.cache_data(ttl=1800)
def load_and_process_data_monthly(url):
    # 创建连接
    conn = st.connection("gsheets", type=GSheetsConnection)

    # 从 Google Sheets 读取数据
    sensor_monthly_source = conn.read(spreadsheet=url, ttl="25m", worksheet="305756283")
    ads_monthly_source = conn.read(spreadsheet=url, ttl="25m", worksheet="1850416648")
    old_new_source = conn.read(spreadsheet=url, ttl="25m", worksheet="666585210")
    spu_index_source = conn.read(spreadsheet=url, ttl="25m", worksheet="455883801")

    # 拿到SPU映射表df，SKU列全大写方便后续映射
    spu_index = pd.DataFrame(spu_index_source)
    spu_index['SKU'] = spu_index['SKU'].str.strip().str.replace('\n', '').replace('\t', '').str.upper()
    # 拿到新老品映射表df，SKU ID列更改表头方便后续映射
    old_new_index = pd.DataFrame(old_new_source)
    old_new_index = old_new_index.rename(columns={'SKU ID': 'SKU'})
    ads_monthly = pd.DataFrame(ads_monthly_source)
    sensor_monthly = pd.DataFrame(sensor_monthly_source)
    sensor_monthly = sensor_monthly.dropna(subset=['SKU'], how='all')
    # 确保 "Month" 列是日期格式
    ads_monthly['Month'] = pd.to_datetime(ads_monthly['Month'])
    sensor_monthly['月'] = pd.to_datetime(sensor_monthly['月'])
    # 定义时间范围
    start_date = pd.to_datetime("2023-01-01")
    end_date = pd.to_datetime("2023-12-01")
    ads_monthly = ads_monthly[(ads_monthly['Month'] >= start_date) & (ads_monthly['Month'] <= end_date)]
    sensor_monthly = sensor_monthly[(sensor_monthly['月'] >= start_date) & (sensor_monthly['月'] <= end_date)]
    return ads_monthly, old_new_index, sensor_monthly,spu_index

#处理ads数据
@st.cache_data(ttl=1800)
def data_process_monthly_ads(df):
    df = df.dropna(subset=['SKU'], how='all')
    # 规则 1: MC ID 为 569301767 时，SKU 去除最后三个字符
    df.loc[df['MC ID'] == 569301767, 'SKU'] = df['SKU'].str[:-3]
    # 规则 2: 处理完MC ID为569301767的数据时，更改MC ID用于后面合并
    df.loc[df['MC ID'] == 569301767, 'MC ID'] = 9174985
    # 规则 3: SKU 带有 "-hm" 时，去除最后三个字符
    df.loc[df['SKU'].str.endswith('-hm',na=False), 'SKU'] = df['SKU'].str[:-3]
    # 规则 4: Currency 为 HKD 时，cost 乘以 0.127 并改变 Currency 为 USD
    df.loc[df['Currency'] == 'HKD', 'cost'] *= 0.127
    df.loc[df['Currency'] == 'HKD', 'ads value'] *= 0.127
    df.loc[df['Currency'] == 'HKD', 'Currency'] = 'USD'
    # 规则 4: 合并重复行
    # 这里假设“合并”是指对于重复的行，我们合并它们的数值列
    newdf = df.groupby(['SKU', 'MC ID', 'Month', 'Currency','Product Type 3','Country'])\
        .agg({'impression': 'sum',
              'cost': 'sum',
              'click': 'sum',
              'conversions': 'sum',
              'ads value': 'sum',
              })\
        .reset_index()
    # 规则 5: 去除不需要的列
    newdf = newdf.drop(columns=['MC ID', 'Currency', 'Country'])
    # 规则 6: SKU列全部大写+去除多余符号以方便匹配
    newdf['SKU'] = newdf['SKU'].str.strip().str.replace('\n', '').replace('\t', '').str.upper()
    return newdf

#合并ads和神策数据
@st.cache_data(ttl=1800)
def get_ads_and_sensor_monthly_merged_data_monthly(ads_monthly, old_new_index, sensor_monthly,spu_index):

    # 先做SPU列映射添加，再做新老品列映射添加
    ads_monthly_merged1 = pd.merge(ads_monthly, spu_index[['SKU','SPU']], on='SKU', how='left')
    # old_new_index[['SKU','customlabel1']]中SKU代表公共列，customlabel1代表要添加的列
    ads_monthly_merged2 = pd.merge(ads_monthly_merged1, old_new_index[['SKU','customlabel1','Sale Price']], on='SKU', how='left')
    # 如果 'customlabel1' 包含 '2023'，则改为 '新品'
    ads_monthly_merged2.loc[ads_monthly_merged2['customlabel1'].str.contains('2023', na=False), 'customlabel1'] = '2023新品'
    # 如果 'customlabel1' 包含 '2022' 或者是空值，则改为 '老品'
    ads_monthly_merged2.loc[ads_monthly_merged2['customlabel1'].str.contains('2022', na=False) | ads_monthly_merged2['customlabel1'].isna(), 'customlabel1'] = '老品'
    ads_monthly_merged2 = ads_monthly_merged2.rename(columns={'customlabel1': 'old_or_new'})

    # 拿到神策月度数据df，做SPU合并
    sensor_monthly_merged1 = pd.merge(sensor_monthly, spu_index[['SKU','SPU']], on='SKU', how='left')
    sensor_monthly_merged2 = pd.merge(sensor_monthly_merged1, old_new_index[['SKU','customlabel1']], on='SKU', how='left')
    sensor_monthly_merged2 = sensor_monthly_merged2.rename(columns={'月': 'Month','销量': 'Sale','加购': 'AddtoCart'})

    # ads和神策数据合并
    ads_and_sensor_monthly_merged = pd.merge(ads_monthly_merged2, sensor_monthly_merged2[['SKU','Month','Sale','GMV','AddtoCart','UV']], on=['SKU', 'Month'], how='left')
    return ads_and_sensor_monthly_merged

#生成最终汇总数据
@st.cache_data(ttl=1800)
def get_summary_df_data(ads_and_sensor_monthly_merged):
    # ads月度数据汇总
    summary_df = ads_and_sensor_monthly_merged.groupby(['SPU', 'SKU','Product Type 3','old_or_new','Sale Price'])\
            .agg({'impression': 'sum',
                  'cost': 'sum',
                  'click': 'sum',
                  'conversions': 'sum',
                  'ads value': 'sum',
                  'Sale': 'sum',
                  'GMV': 'sum',
                  'AddtoCart': 'sum',
                  'UV': 'sum'
                  }).reset_index()
    summary_df = summary_df.sort_values(by='SPU', ascending=True)

    # 添加新的神策ROI计算列
    summary_df['Sensor ROI'] = (summary_df['GMV'] / summary_df['cost'])
    # 处理除以0的情况
    summary_df['Sensor ROI'] = summary_df['Sensor ROI'].fillna(0)  # 将NaN替换为0
    summary_df['Sensor ROI'] = summary_df['Sensor ROI'].replace([np.inf, -np.inf], 0)  # 将无限值替换为0
    # 添加新的adsROI计算列
    summary_df['ads ROI'] = (summary_df['ads value'] / summary_df['cost'])
    # 处理除以0的情况
    summary_df['ads ROI'] = summary_df['ads ROI'].fillna(0)  # 将NaN替换为0
    summary_df['ads ROI'] = summary_df['ads ROI'].replace([np.inf, -np.inf], 0)  # 将无限值替换为0
    # 添加新的点击率计算列
    summary_df['CTR'] = (summary_df['click'] / summary_df['impression'])
    return summary_df

# 新老品样式
@st.cache_data(ttl=1800)
def colorize_old_or_new(val):
    if val == "2023新品":
        color = 'DarkSalmon'
    elif val == "老品":
        color = 'yellow'
    else:  # 其他值的背景保持不变
        color = ''
    return f'background-color: {color}'
# 神策ROI样式
@st.cache_data(ttl=1800)
def colorize_Sensor(val):
    if val < 1.5:
        color = 'LightCoral'
    elif val > 2.5:
        color = 'LightGreen'
    else:  # 其他值的背景保持不变
        color = ''
    return f'background-color: {color}'


ads_monthly, old_new_index, sensor_monthly,spu_index = load_and_process_data_monthly("https://docs.google.com/spreadsheets/d/1bQTrtNC-o9etJ3xUwMeyD8m383xRRq9U7a3Y-gxjP-U/")
# 拿到ads月度数据df，做数据清洗
ads_monthly = data_process_monthly_ads(ads_monthly)
ads_and_sensor_monthly_merged = get_ads_and_sensor_monthly_merged_data_monthly(ads_monthly, old_new_index, sensor_monthly,spu_index)
summary_df = get_summary_df_data(ads_and_sensor_monthly_merged)


tab1, tab2 = st.tabs(["年度数据查询", "SPU登记表"])
with tab1:
    # 在Streamlit中创建多选框，列出DataFrame中的所有列
    st.subheader("SPU所属SKU-2023年度数据查看")

    # 检查会话状态中是否已经有tags，如果没有，则初始化为空列表
    # if 'tags' not in st.session_state:
    #     st.session_state.tags = []

    tags = st_tags(
        label='输入SPU:',
        text='Press enter to add more',
        # value=st.session_state.tags  # 使用会话状态中保存的tags
    )
    # 更新会话状态中的tags
    # st.session_state.tags = tags

    if tags:
        filtered_df = summary_df[summary_df['SPU'].isin(tags)].sort_values(by='cost', ascending=False)
        producttype3 = summary_df[summary_df['SPU'].isin(tags)]['Product Type 3'].iloc[0].upper()
        options = st.multiselect(
            '选择数据维度',
            summary_df.columns,
            ['SPU', 'SKU', 'old_or_new', 'Sale Price', 'cost', 'GMV', 'Sale', 'ads value', 'Sensor ROI', 'ads ROI', 'CTR']
        )
        # 创建一个样式化的DataFrame
        styled_df = filtered_df[options].style

        # 如果用户选择了"old_or_new"列，应用对应的样式
        if 'old_or_new' in options:
            styled_df = styled_df.applymap(colorize_old_or_new, subset=['old_or_new'])

        # 如果用户选择了"Sensor ROI"列，应用对应的样式
        if 'Sensor ROI' in options:
            styled_df = styled_df.applymap(colorize_Sensor, subset=['Sensor ROI'])

        # 应用格式化
        styled_df = styled_df.format({
            'cost': '{:.2f}',
            'GMV': '{:.2f}',
            'Sale Price': '{:.2f}',
            'Sale': '{}',
            'ads value': '{:.2f}',
            'Sensor ROI': '{:.2f}',
            'ads ROI': '{:.2f}',
            'CTR': '{:.2%}',
        })
        st.dataframe(styled_df, width=1500)
        st.write("所属三级类目：",producttype3)
    else:
        st.dataframe(summary_df)
note_url = 'https://docs.google.com/spreadsheets/d/1ZqP2C9a7NG5ksJf_5RyIbIQhVqmFygr0FbsTrJPSqLw/edit#gid=0'
with tab2:
 with st.container():
   col1, col2 = st.columns(2)
   with col1:
        st.subheader('新老品SPU表格')
        conn = st.connection("gsheets", type=GSheetsConnection)
        edite_source = conn.read(spreadsheet=note_url,worksheet='0',ttl="1m")
        daily_df = pd.DataFrame(edite_source)
        daily_df = daily_df.dropna(subset=['SPU'], how='all')
        st.dataframe(daily_df,
                     column_config={
                         "图片": st.column_config.ImageColumn(
                             "图片",
                             width="small"
                         )},
                     height=1000,width=800)
   with col2:
       st.subheader('老品SPU表格')
       conn = st.connection("gsheets", type=GSheetsConnection)
       old_edite_source = conn.read(spreadsheet=note_url,worksheet='1130770782',ttl="1m")
       old_daily_df = pd.DataFrame(old_edite_source)
       old_daily_df = daily_df.dropna(subset=['SPU'], how='all')
       st.dataframe(daily_df,
                    column_config={
                        "图片": st.column_config.ImageColumn(
                            "图片",
                            width="small"
                        )},
                    height=1000,width=800)
