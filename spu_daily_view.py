import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
from streamlit_tags import st_tags
import numpy as np
from datetime import datetime, timedelta
url = "https://docs.google.com/spreadsheets/d/1bQTrtNC-o9etJ3xUwMeyD8m383xRRq9U7a3Y-gxjP-U/edit#gid=0"
st.set_page_config(layout="wide",page_title='SPU数据跟踪',page_icon = 'favicon.png')

@st.cache_data(ttl=3600)
def data_process_monthly_ads(df):
    # 规则 1: MC ID 为 569301767 时，SKU 去除最后三个字符
    df.loc[df['MC ID'] == 569301767, 'SKU'] = df['SKU'].str[:-3]
    # 规则 2: 处理完MC ID为569301767的数据时，更改MC ID用于后面合并
    df.loc[df['MC ID'] == 569301767, 'MC ID'] = 9174985
    # 规则 3: SKU 带有 "-hm" 时，去除最后三个字符
    df.loc[df['SKU'].str.endswith('-hm',na=False), 'SKU',] = df['SKU'].str[:-3]
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

@st.cache_data(ttl=3600)
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

    return ads_monthly, old_new_index, sensor_monthly,spu_index

with st.sidebar:
    # 检查会话状态中是否有 'text' 和 'saved_text'，如果没有，初始化它们
    if 'text' not in st.session_state:
        st.session_state['text'] = ""
    if 'saved_text' not in st.session_state:
        st.session_state['saved_text'] = []

    def pass_param():
        # 保存当前文本区域的值到 'saved_text'
        if len(st.session_state['text']) > 0:
            separatedata = st.session_state['text'].split('\n')
            for singedata in separatedata:
                st.session_state['saved_text'].append(singedata)
        else:
            st.session_state['saved_text'].append(st.session_state['text'])
        # 清空文本区域
        st.session_state['text'] = ""

    def clear_area():
        st.session_state['saved_text'] = []

    # 创建文本区域，其内容绑定到 'text'
    input = st.text_area("批量输入SPU(一个SPU一行)", value=st.session_state['text'], key="text", height=10)

    # 创建一个按钮，当按下时调用 on_upper_clicked 函数
    st.button("确定", on_click=pass_param)

    tags = st_tags(
        label='',
        value=st.session_state['saved_text']  # 使用会话状态中保存的tags
    )

    st.button("清空", on_click=clear_area)
# 缓存装饰器应用于数据加载和处理的函数
@st.cache_data(ttl=1800)
def load_and_process_data_daily(url):
    # 创建连接
    conn = st.connection("gsheets", type=GSheetsConnection)

    # 从 Google Sheets 读取数据
    ads_daily_source = conn.read(spreadsheet=url, ttl="25m", worksheet="0")
    sensor_daily_source = conn.read(spreadsheet=url, ttl="25m", worksheet="7630203")
    old_new_source = conn.read(spreadsheet=url, ttl="25m", worksheet="666585210")
    spu_index_source = conn.read(spreadsheet=url, ttl="25m", worksheet="455883801")

    ads_daily = pd.DataFrame(ads_daily_source)
    old_new_index = pd.DataFrame(old_new_source)
    old_new_index = old_new_index.rename(columns={'SKU ID': 'SKU'})
    spu_index = pd.DataFrame(spu_index_source)
    spu_index['SKU'] = spu_index['SKU'].str.strip().str.replace('\n', '').replace('\t', '').str.upper()
    sensor_daily = pd.DataFrame(sensor_daily_source)
    sensor_daily = sensor_daily.dropna(subset=['SKU'], how='all')
    return ads_daily, old_new_index, sensor_daily, spu_index

@st.cache_data(ttl=1800)
def data_process_daily_ads(df):
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
    newdf = df.groupby(['SKU', 'MC ID', 'Date', 'Currency','Product Type 3','Country'])\
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
def get_ads_and_sensor_monthly_merged_data_daily(ads_daily, old_new_index, sensor_daily,spu_index):

    # 先做SPU列映射添加，再做新老品列映射添加
    ads_daily_merged1 = pd.merge(ads_daily, spu_index[['SKU','SPU']], on='SKU', how='left')
    # old_new_index[['SKU','customlabel1']]中SKU代表公共列，customlabel1代表要添加的列
    ads_daily_merged2 = pd.merge(ads_daily_merged1, old_new_index[['SKU','customlabel1','Sale Price']], on='SKU', how='left')
    # 如果 'customlabel1' 包含 '2023'，则改为 '新品'
    ads_daily_merged2.loc[ads_daily_merged2['customlabel1'].str.contains('2023', na=False), 'customlabel1'] = '2023新品'
    # 如果 'customlabel1' 包含 '2022' 或者是空值，则改为 '老品'
    ads_daily_merged2.loc[ads_daily_merged2['customlabel1'].str.contains('2022', na=False) | ads_daily_merged2['customlabel1'].isna(), 'customlabel1'] = '老品'
    ads_daily_merged2 = ads_daily_merged2.rename(columns={'customlabel1': 'old_or_new'})

    # 拿到神策月度数据df，做SPU合并
    sensor_daily_merged1 = pd.merge(sensor_daily, spu_index[['SKU','SPU']], on='SKU', how='left')
    sensor_daily_merged2 = pd.merge(sensor_daily_merged1, old_new_index[['SKU','customlabel1']], on='SKU', how='left')
    sensor_daily_merged2 = sensor_daily_merged2.rename(columns={'日期': 'Date','销量': 'Sale','加购': 'AddtoCart'})

    # ads和神策数据合并
    ads_and_sensor_daily_merged = pd.merge(ads_daily_merged2, sensor_daily_merged2[['SKU','Date','Sale','GMV','AddtoCart','UV']], on=['SKU', 'Date'], how='left')
    return ads_and_sensor_daily_merged

@st.cache_data(ttl=1800)
def process_monthly_data(summary_df):

    summary_df.loc[summary_df['Month'] == '2023-01-01', ['cost', 'click', 'GMV', 'ads value', 'conversions',
                                                        'Sale', 'AddtoCart','impression','UV']] /= 31
    summary_df.loc[summary_df['Month'] == '2023-02-01', ['cost', 'click', 'GMV', 'ads value', 'conversions',
                                                        'Sale', 'AddtoCart','impression','UV']] /= 28
    summary_df.loc[summary_df['Month'] == '2023-03-01', ['cost', 'click', 'GMV', 'ads value', 'conversions',
                                                        'Sale', 'AddtoCart','impression','UV']] /= 31
    summary_df.loc[summary_df['Month'] == '2023-04-01', ['cost', 'click', 'GMV', 'ads value', 'conversions',
                                                        'Sale', 'AddtoCart','impression','UV']] /= 30
    summary_df.loc[summary_df['Month'] == '2023-05-01', ['cost', 'click', 'GMV', 'ads value', 'conversions',
                                                        'Sale', 'AddtoCart','impression','UV']] /= 31
    summary_df.loc[summary_df['Month'] == '2023-06-01', ['cost', 'click', 'GMV', 'ads value', 'conversions',
                                                        'Sale', 'AddtoCart','impression','UV']] /= 30
    summary_df.loc[summary_df['Month'] == '2023-07-01', ['cost', 'click', 'GMV', 'ads value', 'conversions',
                                                        'Sale', 'AddtoCart','impression','UV']] /= 31
    summary_df.loc[summary_df['Month'] == '2023-08-01', ['cost', 'click', 'GMV', 'ads value', 'conversions',
                                                        'Sale', 'AddtoCart','impression','UV']] /= 31
    summary_df.loc[summary_df['Month'] == '2023-09-01', ['cost', 'click', 'GMV', 'ads value', 'conversions',
                                                        'Sale', 'AddtoCart','impression','UV']] /= 30
    summary_df.loc[summary_df['Month'] == '2023-10-01', ['cost', 'click', 'GMV', 'ads value', 'conversions',
                                                        'Sale', 'AddtoCart','impression','UV']] /= 31
    summary_df.loc[summary_df['Month'] == '2023-11-01', ['cost', 'click', 'GMV', 'ads value', 'conversions',
                                                        'Sale', 'AddtoCart','impression','UV']] /= 30
    summary_df.loc[summary_df['Month'] == '2023-12-01', ['cost', 'click', 'GMV', 'ads value', 'conversions',
                                                        'Sale', 'AddtoCart','impression','UV']] /= 31
    summary_df.loc[summary_df['Month'] == '2024-01-01', ['cost', 'click', 'GMV', 'ads value', 'conversions',
                                                        'Sale', 'AddtoCart','impression','UV']] /= 31
    summary_df = summary_df.groupby(['Month']) \
        .agg({
              'cost': 'sum',
              'click': 'sum',
              'GMV': 'sum',
              'ads value': 'sum',
              'conversions': 'sum',
              'Sale': 'sum',
              'AddtoCart': 'sum',
              'impression': 'sum',
              'UV': 'sum'
              })
    # 添加新的神策ROI计算列
    summary_df['神策ROI'] = (summary_df['GMV'] / summary_df['cost'])
    # 处理除以0的情况
    summary_df['神策ROI'] = summary_df['神策ROI'].fillna(0)  # 将NaN替换为0
    summary_df['神策ROI'] = summary_df['神策ROI'].replace([np.inf, -np.inf], 0)  # 将无限值替换为0
    # 添加新的adsROI计算列
    summary_df['ads ROI'] = (summary_df['ads value'] / summary_df['cost'])
    # 处理除以0的情况
    summary_df['ads ROI'] = summary_df['ads ROI'].fillna(0)  # 将NaN替换为0
    summary_df['ads ROI'] = summary_df['ads ROI'].replace([np.inf, -np.inf], 0)  # 将无限值替换为0
    # 添加新的CPC计算列
    summary_df['CPC'] = (summary_df['cost'] / summary_df['click'])
    summary_df['CPC'] = summary_df['CPC'].fillna(0)  # 将NaN替换为0
    summary_df['CPC'] = summary_df['CPC'].replace([np.inf, -np.inf], 0)  # 将无限值替换为0
    # 添加新的点击率计算列
    summary_df['CTR'] = (summary_df['click'] / summary_df['impression'])*100
    # 添加新的神策转化率计算列
    summary_df['神策转化率'] = (summary_df['Sale'] / summary_df['UV'])*100
    # 添加新的加购率计算列
    summary_df['神策加购率'] = (summary_df['AddtoCart'] / summary_df['UV'])*100
    summary_df = summary_df.rename(columns={'cost': '日均费用','click': '日均点击','GMV': '日均GMV','ads value': '日均ads value','conversions': '日均ads转化',
                               'Sale': '日均销量','AddtoCart': '日均加购','impression': '日均展示','UV': '日均UV'})
    summary_df = summary_df[['日均费用','日均点击','日均GMV','日均ads value','神策ROI','ads ROI','日均销量','CPC','CTR','神策转化率','神策加购率','日均展示']]
    return summary_df

@st.cache_data(ttl=1800)
def process_daily_data(summary_df):
    summary_df = summary_df.groupby(['Date']) \
        .agg({
        'cost': 'sum',
        'click': 'sum',
        'GMV': 'sum',
        'ads value': 'sum',
        'conversions': 'sum',
        'Sale': 'sum',
        'AddtoCart': 'sum',
        'impression': 'sum',
        'UV': 'sum'
    })
    # 添加新的神策ROI计算列
    summary_df['神策ROI'] = (summary_df['GMV'] / summary_df['cost'])
    # 处理除以0的情况
    summary_df['神策ROI'] = summary_df['神策ROI'].fillna(0)  # 将NaN替换为0
    summary_df['神策ROI'] = summary_df['神策ROI'].replace([np.inf, -np.inf], 0)  # 将无限值替换为0
    # 添加新的adsROI计算列
    summary_df['ads ROI'] = (summary_df['ads value'] / summary_df['cost'])
    # 处理除以0的情况
    summary_df['ads ROI'] = summary_df['ads ROI'].fillna(0)  # 将NaN替换为0
    summary_df['ads ROI'] = summary_df['ads ROI'].replace([np.inf, -np.inf], 0)  # 将无限值替换为0
    # 添加新的CPC计算列
    summary_df['CPC'] = (summary_df['cost'] / summary_df['click'])
    summary_df['CPC'] = summary_df['CPC'].fillna(0)  # 将NaN替换为0
    summary_df['CPC'] = summary_df['CPC'].replace([np.inf, -np.inf], 0)  # 将无限值替换为0
    # 添加新的点击率计算列
    summary_df['CTR'] = (summary_df['click'] / summary_df['impression'])*100
    # 添加新的神策转化率计算列
    summary_df['神策转化率'] = (summary_df['Sale'] / summary_df['UV'])*100
    # 添加新的加购率计算列
    summary_df['神策加购率'] = (summary_df['AddtoCart'] / summary_df['UV'])*100
    summary_df = summary_df.rename(columns={'cost': '费用','click': '点击','conversions': 'ads转化',
                               'Sale': '销量','AddtoCart': '加购','impression': '展示','UV': 'UV'})
    summary_df = summary_df[['费用','点击','GMV','ads value','神策ROI','ads ROI','销量','CPC','CTR','神策转化率','神策加购率','展示','UV']]
    return summary_df

@st.cache_data(ttl=1800)
def process_daily_trend_data(summary_df):
    summary_df = summary_df.groupby(['SPU','Product Type 3']) \
        .agg({
        'cost': 'sum',
        'click': 'sum',
        'GMV': 'sum',
        'ads value': 'sum',
        'conversions': 'sum',
        'Sale': 'sum',
        'AddtoCart': 'sum',
        'impression': 'sum',
        'UV': 'sum'
    }).reset_index()
    # 添加新的神策ROI计算列
    summary_df['神策ROI'] = (summary_df['GMV'] / summary_df['cost'])
    # 处理除以0的情况
    summary_df['神策ROI'] = summary_df['神策ROI'].fillna(0)  # 将NaN替换为0
    summary_df['神策ROI'] = summary_df['神策ROI'].replace([np.inf, -np.inf], 0)  # 将无限值替换为0
    # 添加新的adsROI计算列
    summary_df['ads ROI'] = (summary_df['ads value'] / summary_df['cost'])
    # 处理除以0的情况
    summary_df['ads ROI'] = summary_df['ads ROI'].fillna(0)  # 将NaN替换为0
    summary_df['ads ROI'] = summary_df['ads ROI'].replace([np.inf, -np.inf], 0)  # 将无限值替换为0
    # 添加新的CPC计算列
    summary_df['CPC'] = (summary_df['cost'] / summary_df['click'])
    summary_df['CPC'] = summary_df['CPC'].fillna(0)  # 将NaN替换为0
    summary_df['CPC'] = summary_df['CPC'].replace([np.inf, -np.inf], 0)  # 将无限值替换为0
    # 添加新的点击率计算列
    summary_df['CTR'] = (summary_df['click'] / summary_df['impression'])*100
    # 添加新的神策转化率计算列
    summary_df['神策转化率'] = (summary_df['Sale'] / summary_df['UV'])*100
    # 添加新的加购率计算列
    summary_df['神策加购率'] = (summary_df['AddtoCart'] / summary_df['UV'])*100
    summary_df = summary_df.rename(columns={'cost': '费用', 'click': '点击', 'conversions': 'ads转化',
                                            'Sale': '销量', 'AddtoCart': '加购', 'impression': '展示', 'UV': 'UV'})
    summary_df = summary_df[
        ['SPU','Product Type 3','费用', '点击', 'GMV', 'ads value', '神策ROI', 'ads ROI', '销量', 'CPC', 'CTR', '神策转化率', '神策加购率','展示','UV']]
    return summary_df


@st.cache_data(ttl=1800)
def daily_trend_spu_to_sku_data(summary_df):
    summary_df = summary_df.groupby(['SKU','SPU','Product Type 3','old_or_new']) \
        .agg({
        'cost': 'sum',
        'click': 'sum',
        'GMV': 'sum',
        'ads value': 'sum',
        'conversions': 'sum',
        'Sale': 'sum',
        'AddtoCart': 'sum',
        'impression': 'sum',
        'UV': 'sum'
    }).reset_index()
    # 添加新的神策ROI计算列
    summary_df['神策ROI'] = (summary_df['GMV'] / summary_df['cost'])
    # 处理除以0的情况
    summary_df['神策ROI'] = summary_df['神策ROI'].fillna(0)  # 将NaN替换为0
    summary_df['神策ROI'] = summary_df['神策ROI'].replace([np.inf, -np.inf], 0)  # 将无限值替换为0
    # 添加新的adsROI计算列
    summary_df['ads ROI'] = (summary_df['ads value'] / summary_df['cost'])
    # 处理除以0的情况
    summary_df['ads ROI'] = summary_df['ads ROI'].fillna(0)  # 将NaN替换为0
    summary_df['ads ROI'] = summary_df['ads ROI'].replace([np.inf, -np.inf], 0)  # 将无限值替换为0
    # 添加新的CPC计算列
    summary_df['CPC'] = (summary_df['cost'] / summary_df['click'])
    summary_df['CPC'] = summary_df['CPC'].fillna(0)  # 将NaN替换为0
    summary_df['CPC'] = summary_df['CPC'].replace([np.inf, -np.inf], 0)  # 将无限值替换为0
    # 添加新的点击率计算列
    summary_df['CTR'] = (summary_df['click'] / summary_df['impression'])*100
    # 添加新的神策转化率计算列
    summary_df['神策转化率'] = (summary_df['Sale'] / summary_df['UV'])*100
    # 添加新的加购率计算列
    summary_df['神策加购率'] = (summary_df['AddtoCart'] / summary_df['UV'])*100
    summary_df = summary_df.rename(columns={'cost': '费用', 'click': '点击', 'conversions': 'ads转化',
                                            'Sale': '销量', 'AddtoCart': '加购', 'impression': '展示', 'UV': 'UV'})
    summary_df = summary_df[
        ['SKU','SPU','Product Type 3','old_or_new','费用', '点击', 'GMV', 'ads value', '神策ROI', 'ads ROI', '销量', 'CPC', 'CTR', '神策转化率', '神策加购率','展示','UV']]
    return summary_df



def create_date_range_average_summary_data(daily_filtered_date_range_df,dayrange):
    daily_filtered_date_range_df = daily_filtered_date_range_df[['impression', 'cost', 'click', 'conversions', 'ads value', 'Sale', 'GMV', 'AddtoCart', 'UV']].sum()
    daily_filtered_date_range_df /= dayrange
    daily_filtered_date_range_df = daily_filtered_date_range_df.to_frame().T
    # 添加新的神策ROI计算列
    daily_filtered_date_range_df['神策ROI'] = (daily_filtered_date_range_df['GMV'] / daily_filtered_date_range_df['cost'])
    # 处理除以0的情况
    daily_filtered_date_range_df['神策ROI'] = daily_filtered_date_range_df['神策ROI'].fillna(0)  # 将NaN替换为0
    daily_filtered_date_range_df['神策ROI'] = daily_filtered_date_range_df['神策ROI'].replace([np.inf, -np.inf], 0)  # 将无限值替换为0
    # 添加新的adsROI计算列
    daily_filtered_date_range_df['ads ROI'] = (daily_filtered_date_range_df['ads value'] / daily_filtered_date_range_df['cost'])
    # 处理除以0的情况
    daily_filtered_date_range_df['ads ROI'] = daily_filtered_date_range_df['ads ROI'].fillna(0)  # 将NaN替换为0
    daily_filtered_date_range_df['ads ROI'] = daily_filtered_date_range_df['ads ROI'].replace([np.inf, -np.inf], 0)  # 将无限值替换为0
    # 添加新的CPC计算列
    daily_filtered_date_range_df['CPC'] = (daily_filtered_date_range_df['cost'] / daily_filtered_date_range_df['click'])
    daily_filtered_date_range_df['CPC'] = daily_filtered_date_range_df['CPC'].fillna(0)  # 将NaN替换为0
    daily_filtered_date_range_df['CPC'] = daily_filtered_date_range_df['CPC'].replace([np.inf, -np.inf], 0)  # 将无限值替换为0
    # 添加新的点击率计算列
    daily_filtered_date_range_df['CTR'] = (daily_filtered_date_range_df['click'] / daily_filtered_date_range_df['impression'])
    # 添加新的神策转化率计算列
    daily_filtered_date_range_df['神策转化率'] = (daily_filtered_date_range_df['Sale'] / daily_filtered_date_range_df['UV'])
    # 添加新的加购率计算列
    daily_filtered_date_range_df['神策加购率'] = (daily_filtered_date_range_df['AddtoCart'] / daily_filtered_date_range_df['UV'])
    daily_filtered_date_range_df = daily_filtered_date_range_df.rename(columns={'cost': '费用','click': '点击','conversions': 'ads转化',
                               'Sale': '销量','AddtoCart': '加购','impression': '展示','UV': 'UV'})
    sum_df = daily_filtered_date_range_df[['费用','点击','GMV','ads value','神策ROI','ads ROI','销量','CPC','CTR','神策转化率','神策加购率','展示','UV']]
    return st.dataframe(sum_df
    .style
    .format({
        '费用': '{:.2f}',
        '点击': '{:.2f}',
        'GMV': '{:.2f}',
        'ads value': '{:.2f}',
        '神策ROI': '{:.2f}',
        'ads ROI': '{:.2f}',
        '销量': '{:.2f}',
        'CPC': '{:.2f}',
        'CTR': '{:.2%}',
        '神策转化率': '{:.2%}',
        '神策加购率': '{:.2%}'
    })
        , height=50, width=2000
    )


def create_date_range_summary_data(daily_filtered_date_range_df):
    daily_filtered_date_range_df = daily_filtered_date_range_df[['impression', 'cost', 'click', 'conversions', 'ads value', 'Sale', 'GMV', 'AddtoCart', 'UV']].sum()
    daily_filtered_date_range_df = daily_filtered_date_range_df.to_frame().T
    # 添加新的神策ROI计算列
    daily_filtered_date_range_df['神策ROI'] = (daily_filtered_date_range_df['GMV'] / daily_filtered_date_range_df['cost'])
    # 处理除以0的情况
    daily_filtered_date_range_df['神策ROI'] = daily_filtered_date_range_df['神策ROI'].fillna(0)  # 将NaN替换为0
    daily_filtered_date_range_df['神策ROI'] = daily_filtered_date_range_df['神策ROI'].replace([np.inf, -np.inf], 0)  # 将无限值替换为0
    # 添加新的adsROI计算列
    daily_filtered_date_range_df['ads ROI'] = (daily_filtered_date_range_df['ads value'] / daily_filtered_date_range_df['cost'])
    # 处理除以0的情况
    daily_filtered_date_range_df['ads ROI'] = daily_filtered_date_range_df['ads ROI'].fillna(0)  # 将NaN替换为0
    daily_filtered_date_range_df['ads ROI'] = daily_filtered_date_range_df['ads ROI'].replace([np.inf, -np.inf], 0)  # 将无限值替换为0
    # 添加新的CPC计算列
    daily_filtered_date_range_df['CPC'] = (daily_filtered_date_range_df['cost'] / daily_filtered_date_range_df['click'])
    daily_filtered_date_range_df['CPC'] = daily_filtered_date_range_df['CPC'].fillna(0)  # 将NaN替换为0
    daily_filtered_date_range_df['CPC'] = daily_filtered_date_range_df['CPC'].replace([np.inf, -np.inf], 0)  # 将无限值替换为0
    # 添加新的点击率计算列
    daily_filtered_date_range_df['CTR'] = (daily_filtered_date_range_df['click'] / daily_filtered_date_range_df['impression'])
    # 添加新的神策转化率计算列
    daily_filtered_date_range_df['神策转化率'] = (daily_filtered_date_range_df['Sale'] / daily_filtered_date_range_df['UV'])
    # 添加新的加购率计算列
    daily_filtered_date_range_df['神策加购率'] = (daily_filtered_date_range_df['AddtoCart'] / daily_filtered_date_range_df['UV'])
    daily_filtered_date_range_df = daily_filtered_date_range_df.rename(columns={'cost': '费用','click': '点击','conversions': 'ads转化',
                               'Sale': '销量','AddtoCart': '加购','impression': '展示','UV': 'UV'})
    sum_df = daily_filtered_date_range_df[['费用','点击','GMV','ads value','神策ROI','ads ROI','销量','CPC','CTR','神策转化率','神策加购率','展示','UV']]
    return st.dataframe(sum_df
    .style
    .format({
        '费用': '{:.2f}',
        '点击': '{:.2f}',
        'GMV': '{:.2f}',
        'ads value': '{:.2f}',
        '神策ROI': '{:.2f}',
        'ads ROI': '{:.2f}',
        '销量': '{:.2f}',
        'CPC': '{:.2f}',
        'CTR': '{:.2%}',
        '神策转化率': '{:.2%}',
        '神策加购率': '{:.2%}'
    })
        , height=50, width=2000
    )

def create_dynamic_column_setting(raw_select_category_df, avoid_list, percentage_list, int_list):
    column_config = {}
    for column in raw_select_category_df.columns:
        # 跳过“SKU”和“SPU”列
        if column in avoid_list:
            continue
        max_value = float(raw_select_category_df[column].max())
        if column in percentage_list:  # 百分比格式
            column_config[column] = st.column_config.ProgressColumn(
                format='%.2f%%',  # 显示为百分比
                min_value=0,
                max_value=max_value,
                label=column
            )
        elif column in int_list:  # 整数格式
            column_config[column] = st.column_config.ProgressColumn(
                format='',  # 显示为整数
                min_value=0,
                max_value=max_value,
                label=column
            )
        else:  # 其它列的正常处理
            column_config[column] = st.column_config.ProgressColumn(
                format='%.2f' if raw_select_category_df[column].dtype == float else '{value}',
                # 浮点数保留两位小数
                min_value=0,
                max_value=max_value,
                label=column
            )
    return column_config

# 定义一个颜色映射函数
@st.cache_data(ttl=1800)
def color_cost(value):
    """
    根据cost的值返回颜色代码。
    值越高，颜色越深。
    """
    if pd.isnull(value) or value == 0:
        return 'background-color: white'
    elif 0 < value < 50:
        return 'background-color: lightgreen'
    elif 50 <= value < 150:
        return 'background-color: lightyellow'
    else:
        return 'background-color: salmon'
def apply_styles(col):
    # 只为非 'SKU' 和 'SPU' 列应用样式
    if col.name not in ['SKU', 'SPU','old_or_new']:
        return col.map(color_cost)
    else:
        return ['' for _ in col]

ads_monthly, old_new_index, sensor_monthly,spu_index = load_and_process_data_monthly(url)
ads_daily, old_new_index, sensor_daily, spu_index = load_and_process_data_daily(url)
ads_monthly = data_process_monthly_ads(ads_monthly)
ads_daily = data_process_daily_ads(ads_daily)
ads_and_sensor_daily_merged = get_ads_and_sensor_monthly_merged_data_daily(ads_daily, old_new_index, sensor_daily,spu_index)
ads_and_sensor_monthly_merged = get_ads_and_sensor_monthly_merged_data_monthly(ads_monthly, old_new_index, sensor_monthly,spu_index)

tab1, tab2, tab3 = st.tabs(["SPU日维度趋势(谷歌CPC渠道GMV,无细分)","SPU查询","SPU登记表"])

with tab2:
    if tags:
        monthly_filtered_df = ads_and_sensor_monthly_merged[ads_and_sensor_monthly_merged['SPU'].isin(tags)]
        monthly_summary_df = process_monthly_data(monthly_filtered_df)
        column_config = create_dynamic_column_setting(monthly_summary_df, ['Month'], ['CTR','神策转化率','神策加购率'], [])
        st.text("查询SPU历史月度日均数据")
        st.dataframe(monthly_summary_df,
        column_config= column_config,
        height=300,width=2000)
        daily_filtered_df = ads_and_sensor_daily_merged[ads_and_sensor_daily_merged['SPU'].isin(tags)]
        daily_summary_df = process_daily_data(daily_filtered_df)

        min_date = ads_daily['Date'].min()
        min_date = datetime.strptime(min_date, "%Y-%m-%d")
        max_date = ads_daily['Date'].max()
        max_date = datetime.strptime(max_date, "%Y-%m-%d")
        default_start_date = datetime.today() - timedelta(days=7)
        default_end_date = datetime.today() - timedelta(days=1)


        selected_range = st.date_input(
            "日均数据自定义日期范围",
            [default_start_date, default_end_date],
            min_value=min_date,
            max_value=max_date
        )
        if len(selected_range) == 2:
            daily_filtered_df['Date'] = pd.to_datetime(daily_filtered_df['Date'])
            daily_filtered_date_range_df = daily_filtered_df[(daily_filtered_df['Date'] >= pd.to_datetime(selected_range[0])) & (daily_filtered_df['Date'] <= pd.to_datetime(selected_range[1]))]
            days_difference = (selected_range[1] - selected_range[0]).days
            st.text("自定义日期日均数据")
            create_date_range_average_summary_data(daily_filtered_date_range_df,days_difference)
            st.text("自定义日期汇总数据")
            create_date_range_summary_data(daily_filtered_date_range_df)
        column_config = create_dynamic_column_setting(daily_summary_df, ['Date'],
                                                      ['神策加购率', '神策转化率', 'CTR'], ['UV', '点击', '展示','销量'])
        st.text("查询SPU日维度明细")
        st.dataframe(daily_summary_df,
        column_config = column_config,
        height=300,width=2000)
        with st.container():
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("查询SPU月维度数据历史趋势")
                options = st.multiselect(
                    '选择数据维度',
                    monthly_summary_df.columns,
                    ['日均费用','日均GMV']
                )
                st.line_chart(monthly_summary_df[options])
            with col2:
                st.subheader("查询SPU日维度数据历史趋势")
                options = st.multiselect(
                    '选择数据维度',
                    daily_summary_df.columns,
                    ['费用','点击']
                )
                st.line_chart(daily_summary_df[options])

        daily_cost_trend_df = ads_and_sensor_daily_merged[ads_and_sensor_daily_merged['SPU'].isin(tags)]
        pivot_df = daily_cost_trend_df.pivot(index=['SPU','SKU','old_or_new'], columns='Date', values='cost').reset_index()
        # 提取SKU列的唯一值
        unique_sku = pivot_df['SPU'].unique()
        st.subheader('查询SPU日维度SKU花费分布')
        options = st.multiselect(
            '选择SPU',
            unique_sku,
            [unique_sku[0]]
        )
        if options:
         st.dataframe(pivot_df[pivot_df['SPU'].isin(options)]
                         .style
                         .apply(apply_styles, axis=0)
                         .format(
                         {col: "{:.2f}" for col in pivot_df.columns if col != 'SKU' and col != 'SPU' and col != 'old_or_new'}
                         )
                         ,height=500,width=2000)
        else:
            st.dataframe(pivot_df
            .style
            .apply(apply_styles, axis=0)
            .format(
                {col: "{:.2f}" for col in pivot_df.columns if col != 'SKU' and col != 'SPU' and col != 'old_or_new'}
            )
                , height=500,width=2000)
    else:
      st.write("请输入SPU")

with tab1:

    min_date = ads_daily['Date'].min()
    min_date = datetime.strptime(min_date, "%Y-%m-%d")
    max_date = ads_daily['Date'].max()
    max_date = datetime.strptime(max_date, "%Y-%m-%d")
    default_start_date = datetime.today() - timedelta(days=7)
    default_end_date = datetime.today() - timedelta(days=1)
    daily_trend_selected_range = st.date_input(
        "日趋势日期范围",
        [default_start_date, default_end_date],
        min_value=min_date,
        max_value=max_date
    )

    daily_filtered_df = ads_and_sensor_daily_merged
    # 选择日期范围内的数据
    ads_and_sensor_daily_merged['Date'] = pd.to_datetime(ads_and_sensor_daily_merged['Date'])
    # 处理普通选择日期范围内的数据
    ads_seonsor_daily_filtered_date_range_df = ads_and_sensor_daily_merged[(ads_and_sensor_daily_merged['Date'] >= pd.to_datetime(daily_trend_selected_range[0])) & (ads_and_sensor_daily_merged['Date'] <= pd.to_datetime(daily_trend_selected_range[1]))]
    filtered_date_range_summary_df = process_daily_trend_data(ads_seonsor_daily_filtered_date_range_df)
    trend_summary_options = st.multiselect(
        '选择数据维度',
        filtered_date_range_summary_df.columns,
        ['SPU','Product Type 3','费用', '点击','GMV','神策ROI','ads ROI']
    )
    unique_category = filtered_date_range_summary_df['Product Type 3'].unique()
    category_options = st.multiselect(
        '选择三级类目',
        unique_category
    )
    if category_options:
        raw_select_category_df = filtered_date_range_summary_df[trend_summary_options]
        select_category_df = raw_select_category_df[raw_select_category_df['Product Type 3'].isin(category_options)]
        column_config = create_dynamic_column_setting(raw_select_category_df, ['SPU', 'Product Type 3'],
                                                      ['神策加购率', '神策转化率', 'CTR'], ['UV', '点击', '展示'])
        st.text('SPU日趋势')
        st.dataframe(select_category_df,
                     column_config=column_config,
                     height=400, width=2000)

    else:
        raw_select_category_df = filtered_date_range_summary_df[trend_summary_options]
        column_config = create_dynamic_column_setting(raw_select_category_df,['SPU', 'Product Type 3'],['神策加购率', '神策转化率', 'CTR'],['UV','点击','展示'])
        st.text('SPU日趋势')
        st.dataframe(raw_select_category_df,
         column_config=column_config,
         height=400, width=2000)
    if tags:
        ads_seonsor_daily_filtered_date_range_df = ads_seonsor_daily_filtered_date_range_df[ads_seonsor_daily_filtered_date_range_df['SPU'].isin(tags)]
        spu_sku_df = daily_trend_spu_to_sku_data(ads_seonsor_daily_filtered_date_range_df)
        spu_sku_summary_options = st.multiselect(
            '选择SKU细分数据维度',
            spu_sku_df.columns,
            ['SKU','SPU','Product Type 3','old_or_new','费用', '点击', 'GMV','神策ROI','ads ROI','神策转化率','神策加购率']
        )
        column_config = create_dynamic_column_setting(spu_sku_df[spu_sku_summary_options], ['SPU','SKU', 'Product Type 3','old_or_new'],
                                                      ['神策加购率', '神策转化率', 'CTR'], ['UV', '点击', '展示'])
        st.text('细分SKU趋势')
        st.dataframe(spu_sku_df[spu_sku_summary_options],
        column_config=column_config,
        height=300, width=2000)
note_url = 'https://docs.google.com/spreadsheets/d/1ZqP2C9a7NG5ksJf_5RyIbIQhVqmFygr0FbsTrJPSqLw/edit#gid=0'
with tab3:
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
       st.dataframe(old_daily_df,
                     column_config={
                         "图片": st.column_config.ImageColumn(
                             "图片",
                             width="small"
                         )},
                     height=1000,width=800)
