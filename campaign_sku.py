import pandas as pd
import streamlit as st
from streamlit_gsheets import GSheetsConnection
from streamlit_tags import st_tags
from datetime import datetime, timedelta
import numpy as np
st.set_page_config(layout="wide",page_title='广告系列')
@st.cache_data(ttl=2400)
def load_and_process_data():
    # 创建连接
    ads_url = 'https://docs.google.com/spreadsheets/d/13XPznGvetWodShbQouOiJVueqVYl5d0e5wWORDTRXoQ/edit#gid=0'
    sensor_url = 'https://docs.google.com/spreadsheets/d/1dN0gfmPcURMh34vqKq0A_l8zYuHwg0zW0-lZ8dQNRUw/edit#gid=0'
    conn = st.connection("gsheets", type=GSheetsConnection)
    old_new_source = conn.read(spreadsheet=ads_url, ttl="25m", worksheet="561998802")
    spu_index_source = conn.read(spreadsheet=ads_url, ttl="25m", worksheet="613856799")
    ads_daily_source = conn.read(spreadsheet=ads_url, ttl="25m", worksheet="0")
    sensor_daily_source = conn.read(spreadsheet=sensor_url, ttl="2m", worksheet="0")
    old_new_df = pd.DataFrame(old_new_source)
    old_new_df = old_new_df.rename(columns={'SKU ID': 'SKU'})
    spu_index_df = pd.DataFrame(spu_index_source)
    spu_index_df['SKU'] = spu_index_df['SKU'].str.strip().str.replace('\n', '').replace('\t', '').str.upper()
    ads_daily_df = pd.DataFrame(ads_daily_source)
    sensor_daily_df = pd.DataFrame(sensor_daily_source)
    sensor_daily_df = sensor_daily_df.rename(columns={'行为时间': 'Date'})
    return sensor_daily_df,ads_daily_df,old_new_df,spu_index_df

@st.cache_data(ttl=2400)
def data_process_daily_ads(df):
    df = df.dropna(subset=['SKU ID'], how='all')
    df = df.rename(columns={'SKU ID': 'SKU'})
    # 规则 1: MC ID 为 569301767 时，SKU 去除最后三个字符
    df.loc[df['MC ID'] == 569301767, 'SKU'] = df['SKU'].str[:-3]
    # 规则 2: 处理完MC ID为569301767的数据时，更改MC ID用于后面合并
    df.loc[df['MC ID'] == 569301767, 'MC ID'] = 9174985
    # 规则 3: SKU 带有 "-hm" 时，去除最后三个字符
    df.loc[df['SKU'].str.endswith('-hm',na=False), 'SKU'] = df['SKU'].str[:-3]
    # 规则 4: Currency 为 HKD 时，cost 乘以 0.127 并改变 Currency 为 USD
    df.loc[df['currency'] == 'HKD', 'cost'] *= 0.127
    df.loc[df['currency'] == 'HKD', 'value'] *= 0.127
    df.loc[df['currency'] == 'HKD', 'currency'] = 'USD'
    # 规则 4: 合并重复行
    # 这里假设“合并”是指对于重复的行，我们合并它们的数值列
    newdf = df.groupby(['Campaign','Campaign ID','MC ID','SKU','Date', 'currency','Product Type 3','Country'])\
        .agg({'Impression': 'sum',
              'cost': 'sum',
              'clicks': 'sum',
              'conversion': 'sum',
              'value': 'sum',
              })\
        .reset_index()
    # 规则 5: 去除不需要的列
    newdf = newdf.drop(columns=['MC ID', 'currency', 'Country'])
    # 规则 6: SKU列全部大写+去除多余符号以方便匹配
    newdf['SKU'] = newdf['SKU'].str.strip().str.replace('\n', '').replace('\t', '').str.upper()
    return newdf

@st.cache_data(ttl=2400)
def merged_spu_and_oldnew_image_data(ads_daily, old_new_index,spu_index):
    # 先做SPU列映射添加，再做新老品列映射添加
    daily_merged1 = pd.merge(ads_daily, spu_index[['SKU', 'SPU']], on='SKU', how='left')
    # old_new_index[['SKU','customlabel1']]中SKU代表公共列，customlabel1代表要添加的列
    daily_merged2 = pd.merge(daily_merged1, old_new_index[['SKU', 'customlabel1', 'Sale Price','imagelink']], on='SKU',
                                 how='left')
    # 如果 'customlabel1' 包含 '2023'，则改为 '新品'
    daily_merged2.loc[daily_merged2['customlabel1'].str.contains('2023', na=False), 'customlabel1'] = '2023新品'
    # 如果 'customlabel1' 包含 '2022' 或者是空值，则改为 '老品'
    daily_merged2.loc[daily_merged2['customlabel1'].str.contains('2022', na=False) | daily_merged2[
        'customlabel1'].isna(), 'customlabel1'] = '老品'
    ads_daily_merged2 = daily_merged2.rename(columns={'customlabel1': 'old_or_new','imagelink': 'image'})
    return ads_daily_merged2

@st.cache_data(ttl=2400)
def create_dynamic_column_setting(raw_select_df, avoid_list, image_list,barlist):
    column_config = {}
    for column in raw_select_df.columns:
        # 跳过“SKU”和“SPU”列
        if column in avoid_list:
            continue
        if column in image_list:  # 百分比格式
            column_config[column] = st.column_config.ImageColumn(
                width="small"
            )
        elif column in barlist:  # 百分比格式
            max_value = float(raw_select_df[column].max())
            column_config[column] = st.column_config.ProgressColumn(
                format='%.2f',  # 显示为百分比
                min_value=0,
                max_value=max_value,
                label=column
            )
        else:  # 其它列的正常处理
            column_config[column] = st.column_config.BarChartColumn(
                width='small'
            )
    return column_config

@st.cache_data(ttl=2400)
def create_summary_data(df):
    # 添加新的CPC计算列
    df['CPC'] = (df['cost'] / df['clicks'])
    df['CPC'] = df['CPC'].fillna(0)  # 将NaN替换为0
    df['CPC'] = df['CPC'].replace([np.inf, -np.inf], 0)  # 将无限值替换为0
    cost_sum = df.groupby(
        ['SKU', 'SPU', 'Product Type 3', 'old_or_new', 'Sale Price', 'image'])['cost'].sum().rename(
        'cost_sum').reset_index()
    gmv_sum = df.groupby(
        ['SKU', 'SPU', 'Product Type 3', 'old_or_new', 'Sale Price', 'image'])['GMV'].sum().rename(
        'gmv_sum').reset_index()
    # 把带日期的合并了
    grouped_df = df.groupby(['SKU', 'SPU', 'Date','Product Type 3', 'old_or_new', 'Sale Price', 'image','CPC']) \
        .agg(
        {'cost': 'sum', 'clicks': 'sum', 'Impression': 'sum', 'GMV': 'sum', 'UV': 'sum', 'value': 'sum', 'conversion': 'sum',
         'sale': 'sum', 'saleuser': 'sum', 'AddtoCart': 'sum'}) \
        .reset_index()
    # 再处理成list
    grouped_df['Date'] = pd.to_datetime(grouped_df['Date'])

    grouped_df['Date'] = grouped_df['Date'].dt.strftime('%Y-%m-%d')
    grouped_df = grouped_df.groupby(['SKU','SPU','Product Type 3','old_or_new','Sale Price','image'])\
        .agg({'CPC': list,'Date': list,'cost': list,'clicks': list,'Impression': list, 'GMV': list,'UV': list,'value': list,'conversion': list,'sale': list,'saleuser': list,'AddtoCart':list})\
        .reset_index()

    grouped_df = pd.merge(grouped_df, gmv_sum,
                          on=['SKU', 'SPU', 'Product Type 3', 'old_or_new', 'Sale Price', 'image'],
                          how='left')
    grouped_df = pd.merge(grouped_df, cost_sum,
                          on=['SKU', 'SPU', 'Product Type 3', 'old_or_new', 'Sale Price', 'image'],
                          how='left')

    return grouped_df


sensor_daily_df,ads_daily_df,old_new_df,spu_index_df = load_and_process_data()
process_ads_daily_df  = data_process_daily_ads(ads_daily_df)
merged_spu_and_oldnew_ads_daily_df = merged_spu_and_oldnew_image_data(process_ads_daily_df, old_new_df,spu_index_df)
merged_spu_and_oldnew_ads_daily_df['Date'] = pd.to_datetime(merged_spu_and_oldnew_ads_daily_df['Date'])
sensor_daily_df['Date'] = pd.to_datetime(sensor_daily_df['Date'])
ads_merged_sensor_data_daily = pd.merge(merged_spu_and_oldnew_ads_daily_df,sensor_daily_df[['Date','Campaign','Campaign ID','SKU','saleuser','GMV','UV','sale','AddtoCart']], on=['Date','Campaign','Campaign ID','SKU'], how='left')
# sensor_merged_ads_data_daily = pd.merge(sensor_daily_df,merged_spu_and_oldnew_ads_daily_df[['Date','Campaign','Campaign ID','SKU','Product Type 3','Impression','cost','clicks','conversion','value','SPU','old_or_new','Sale Price','image']], on=['Date','Campaign','Campaign ID','SKU'], how='left')
ads_merged_sensor_data_daily['Date'] = pd.to_datetime(ads_merged_sensor_data_daily['Date'])
ads_merged_sensor_data_daily.sort_values(by=['SKU', 'Date'], ascending=[True, False], inplace=True)
ads_merged_sensor_data_daily = ads_merged_sensor_data_daily.fillna(0)

ads_daily_df['Date'] = pd.to_datetime(ads_daily_df['Date'])
min_date = ads_daily_df['Date'].min()
max_date = ads_daily_df['Date'].max()
default_start_date = datetime.today() - timedelta(days=14)
default_end_date = datetime.today() - timedelta(days=7)
selected_range = st.date_input(
    "自定义日期范围",
    [default_start_date, default_end_date],
    min_value=min_date,
    max_value=max_date
)

ads_merged_sensor_data_daily['Date'] = pd.to_datetime(ads_merged_sensor_data_daily['Date'])
daily_filtered_date_range_df = ads_merged_sensor_data_daily[(ads_merged_sensor_data_daily['Date'] >= pd.to_datetime(selected_range[0])) & (ads_merged_sensor_data_daily['Date'] <= pd.to_datetime(selected_range[1]))]

with st.sidebar:
    # 检查会话状态中是否有 'text' 和 'saved_text'，如果没有，初始化它们
    if 'campaign_text' not in st.session_state:
        st.session_state['campaign_text'] = ""
    if 'campaign_saved_text' not in st.session_state:
        st.session_state['campaign_saved_text'] = []
    def pass_param():
        # 保存当前文本区域的值到 'saved_text'
        if len(st.session_state['campaign_text']) > 0:
            separatedata = st.session_state['campaign_text'].split('\n')
            for singedata in separatedata:
                st.session_state['campaign_saved_text'].append(singedata)
        else:
            st.session_state['campaign_saved_text'].append(st.session_state['campaign_text'])
        # 清空文本区域
        st.session_state['campaign_text'] = ""
    def clear_area():
        st.session_state['campaign_saved_text'] = []
    # 创建文本区域，其内容绑定到 'text'
    input = st.text_area("批量输入广告系列名(一个广告系列名一行)", value=st.session_state['campaign_text'], key="campaign_text", height=10)
    # 创建一个按钮，当按下时调用 on_upper_clicked 函数
    st.button("确定", on_click=pass_param)
    tags = st_tags(
        label='',
        value=st.session_state['campaign_saved_text']  # 使用会话状态中保存的tags
    )
    st.button("清空", on_click=clear_area)
# 批量框输入系列
if tags:
    campaign_filtered_df = daily_filtered_date_range_df[daily_filtered_date_range_df['Campaign'].isin(tags)]
    grouped_df = create_summary_data(campaign_filtered_df)
    column_config = create_dynamic_column_setting(grouped_df, ['SKU','SPU','Product Type 3','old_or_new','Sale Price','Date'],['image'],['cost_sum','gmv_sum'])
    options = st.multiselect(
        '选择数据维度',
        grouped_df.columns,
        ['SKU', 'Product Type 3', 'image', 'cost_sum', 'gmv_sum', 'Impression','CPC','cost', 'clicks']
    )
    st.dataframe(grouped_df[options],
    column_config=column_config,
    width=1600,height=400)
# 批量框外的情况
else:
    unique_camapign = daily_filtered_date_range_df['Campaign'].unique()
    campaign_options = st.multiselect(
        '广告系列',
        unique_camapign)
    # 自选框中输入系列
    if campaign_options:
       select_campaign_df =  daily_filtered_date_range_df[daily_filtered_date_range_df['Campaign'].isin(campaign_options)]
       grouped_df = create_summary_data(select_campaign_df)
       column_config = create_dynamic_column_setting(grouped_df,
                                                   ['SKU', 'SPU', 'Product Type 3', 'old_or_new',
                                                    'Sale Price','Date'], ['image'], ['cost_sum', 'gmv_sum'])
       options = st.multiselect(
            '选择数据维度',
            grouped_df.columns,
            ['SKU', 'Product Type 3', 'image', 'cost_sum', 'gmv_sum', 'Impression','CPC','cost', 'clicks']
        )
       st.dataframe(grouped_df[options],
         column_config=column_config,
         width=1600,height=400)
    # 自选框默认留空
    else:
      grouped_df = create_summary_data(daily_filtered_date_range_df)
      column_config = create_dynamic_column_setting(grouped_df,
                                                    ['SKU', 'SPU', 'Product Type 3', 'old_or_new',
                                                     'Sale Price','Date'], ['image'], ['cost_sum', 'gmv_sum'])
      options = st.multiselect(
        '选择数据维度',
        grouped_df.columns,
        ['SKU', 'Product Type 3', 'image', 'cost_sum', 'gmv_sum', 'Impression','CPC','cost', 'clicks']
    )
      st.dataframe(grouped_df[options],
                 column_config=column_config,
                 width=1600, height=400)
