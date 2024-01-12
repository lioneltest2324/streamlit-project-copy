import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
import re
import numpy as np
from datetime import datetime, timedelta
from functools import reduce
import operator
from streamlit_tags import st_tags
sensor_url = 'https://docs.google.com/spreadsheets/d/1X0YPC6iAZn1Lu4szX67fi5h4B8HiVbfA-i68EyzpOq0/edit#gid=0'
ads_url = 'https://docs.google.com/spreadsheets/d/13G1sZWVLKa_kpScqGVmNp-5abCTkxmAFW0dxW29DMUY/edit#gid=0'
spu_index_url = "https://docs.google.com/spreadsheets/d/1bQTrtNC-o9etJ3xUwMeyD8m383xRRq9U7a3Y-gxjP-U/edit#gid=0"
st.set_page_config(layout="wide")
@st.cache_data(ttl=2400)
def load_and_process_data(url,worksheetname):
    # 创建连接
    conn = st.connection("gsheets", type=GSheetsConnection)
    # 从 Google Sheets 读取数据
    source = conn.read(spreadsheet=url, ttl="30m", worksheet=worksheetname)
    daily_df = pd.DataFrame(source)
    daily_df = daily_df.dropna(thresh=len(daily_df.columns) - 2)
    return daily_df

@st.cache_data(ttl=2400)
def data_process_daily_ads(df):
    # 规则 1: MC ID 为 569301767 时，SKU 去除最后三个字符
    df.loc[df['MC ID'] == 569301767, 'SKU'] = df['SKU'].str[:-3]
    # 规则 2: 处理完MC ID为569301767的数据时，更改MC ID用于后面合并
    df.loc[df['MC ID'] == 569301767, 'MC ID'] = 9174985
    # 规则 3: SKU 带有 "-hm" 时，去除最后三个字符
    df.loc[df['SKU'].str.endswith('-hm',na=False), 'SKU',] = df['SKU'].str[:-3]
    # 规则 4: Currency 为 HKD 时，cost 乘以 0.127 并改变 Currency 为 USD
    df.loc[df['Currency'] == 'HKD', 'cost'] *= 0.13
    df.loc[df['Currency'] == 'HKD', 'ads value'] *= 0.13
    df.loc[df['Currency'] == 'HKD', 'Currency'] = 'USD'
    # 规则 4: 处理新老品
    # 如果 'customlabel1' 包含 '2023'，则改为 '新品'
    df.loc[df['customlabel1'].str.contains('2023', na=False), 'customlabel1'] = '2023新品'
    # 如果 'customlabel1' 包含 '2022' 或者是空值，则改为 '老品'
    df.loc[df['customlabel1'].str.contains('2022', na=False) | df['customlabel1'].isna(), 'customlabel1'] = '老品'
    df = df.rename(columns={'customlabel1': 'old_or_new'})
    # 规则 5: 合并重复行
    # 这里假设“合并”是指对于重复的行，我们合并它们的数值列
    newdf = df.groupby(['SKU', 'MC ID', 'Date', 'Currency','Product Type 1','Product Type 2','Product Type 3','old_or_new','Country'])\
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

@st.cache_data(ttl=2400)
def get_spu_merged_data(ads_daily, spu_index):
    # 先做SPU列映射添加，再做新老品列映射添加
    ads_monthly_merged1 = pd.merge(ads_daily, spu_index[['SKU', 'SPU']], on='SKU', how='left')
    return ads_monthly_merged1

@st.cache_data(ttl=2400)
def create_sensor_summary_data(seonsor_daily):
    summary_df = seonsor_daily.groupby(['SKU','SPU']) \
        .agg({
        'saleuser': 'sum',
        'sale': 'sum',
        'GMV': 'sum',
        'AddtoCart': 'sum',
        'UV': 'sum'
    }).reset_index()
    # 添加新的神策转化率计算列
    summary_df['神策转化率'] = (summary_df['saleuser'] / summary_df['UV'])
    # 添加新的加购率计算列
    summary_df['神策加购率'] = (summary_df['AddtoCart'] / summary_df['UV'])
    return summary_df

def sensor_and_ads_merged_data(sensor_summary_data,ads_daily_merged_spu_data):
    # 这里不用groupby的话会直接套用ads_daily_merged_spu_data的全数据
    merged_data1 = pd.merge(ads_daily_merged_spu_data, sensor_summary_data[['SKU', 'SPU','saleuser','sale','GMV','UV','AddtoCart','神策转化率','神策加购率']], on=['SKU','SPU'], how='left')
    summary_df = merged_data1.groupby(['SKU', 'SPU','saleuser','sale','GMV','UV','AddtoCart','Product Type 1','Product Type 2','Product Type 3','old_or_new','神策加购率','神策转化率']) \
        .agg({
        'impression': 'sum',
              'cost': 'sum',
              'click': 'sum',
              'conversions': 'sum',
              'ads value': 'sum',
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
    summary_df['CTR'] = (summary_df['click'] / summary_df['impression'])
    summary_df = summary_df.rename(columns={'cost': '费用', 'click': '点击', 'conversions': 'ads转化',
                 'sale': '销量', 'saleuser': '支付用户数','AddtoCart': '加购', 'impression': '展示'})
    return summary_df

@st.cache_data(ttl=2400)
def summary_df_merged_customlabel(summary_df,ads_daily):
    summary_df_merged_customlabel = pd.merge(summary_df,ads_daily[['SKU','customlabel2','customlabel4']],on='SKU',how='left')
    summary_df_merged = summary_df_merged_customlabel.groupby(['SPU', 'SKU', '费用', 'GMV', '点击', '支付用户数', '神策ROI', 'ads ROI','customlabel2','customlabel4','Product Type 1','Product Type 2','Product Type 3','old_or_new','神策加购率','神策转化率']).first().reset_index()
    return summary_df_merged

@st.cache_data(ttl=2400)
def return_combine_all_summary_df(summary_df,selected_range):
    summary_df = summary_df.drop(columns=['神策加购率', '神策转化率', '神策ROI','ads ROI','CTR','CPC','Product Type 1','Product Type 2','Product Type 3','old_or_new','SPU', 'SKU'])
    summary_group_df = summary_df.sum().to_frame().T
    summary_group_df['日期范围'] = selected_range[0].strftime('%Y-%m-%d')+"至"+selected_range[1].strftime('%Y-%m-%d')
    summary_group_df['神策转化率'] = (summary_group_df['支付用户数'] / summary_group_df['UV'])
    summary_group_df['神策加购率'] = (summary_group_df['加购'] / summary_group_df['UV'])
    summary_group_df['点击率'] = (summary_group_df['点击'] / summary_group_df['展示'])
    summary_group_df['ads ROI'] = (summary_group_df['ads value'] / summary_group_df['费用'])
    summary_group_df['神策 ROI'] = (summary_group_df['GMV'] / summary_group_df['费用'])
    summary_group_df['CPC'] = (summary_group_df['费用'] / summary_group_df['点击'])
    date_forward = ['日期范围'] + [col for col in summary_group_df.columns if col != '日期范围']
    summary_group_df = summary_group_df[date_forward]
    # compare_summary_group_df = compare_summary_df.groupby(['费用', 'GMV', '点击', '支付用户数', '神策ROI', 'ads ROI','神策加购率','神策转化率','ads转化','ads value','CPC','UV','加购','展示','CTR']).first().reset_index()
    merged_df = summary_group_df
    return merged_df

# 汇总列百分比处理
@st.cache_data(ttl=2400)
def format_comparison(row):
    if row['日期范围'] == '对比':
        # 只有当 '日期范围' 列的值是 '对比' 时，才进行格式化
        return [f"{x*100:.2f}%" if isinstance(x, (int, float)) and col != '日期范围' else x for col, x in row.iteritems()]
    else:
        return row  # 对于不是 '对比' 的行，返回原始行

# 汇总列样式处理
@st.cache_data(ttl=2400)
def colorize_comparison(row):
    # 新建一个与行长度相同的样式列表，初始值为空字符串
    colors = [''] * len(row)
    # 检查当前行是否为 "对比" 行
    if row['日期范围'] == '对比':
        # 遍历除了 '日期范围' 列之外的所有值
        for i, v in enumerate(row):
            # 跳过 '日期范围' 列
            if row.index[i] != '日期范围':
                try:
                    # 将字符串转换为浮点数进行比较
                    val = float(v.strip('%'))
                    if val <= 0:
                        colors[i] = 'background-color: LightCoral'
                    elif val >= 0:
                        colors[i] = 'background-color: LightGreen'
                except ValueError:
                    # 如果转换失败，说明这不是一个数值，忽略它
                    pass
    # 返回颜色样式列表
    return colors

# 自定义格式化函数
@st.cache_data(ttl=2400)
def format_first_two_rows(value, format_str):
    if pd.notna(value):
        return format_str.format(value)
    return value

@st.cache_data(ttl=2400)
def create_compare_summary_df(origin_df,compare_df):
    # 合并 DataFrame
    combined_df = pd.concat([origin_df, compare_df], ignore_index=True)
    # 计算百分比变化
    comparison = {}
    for col in origin_df.columns:
        if col != '日期范围':
            val1 = origin_df[col].values[0]
            val2 = compare_df[col].values[0]
            # 计算百分比变化
            if val1 != 0:  # 防止除以零
                change = ((val2 - val1) / val1)
            else:
                change = ''  # 如果原值为0，则变化无穷大
            comparison[col] = change
    # 添加对比行
    comparison['日期范围'] = '对比'
    combined_df = combined_df.append(comparison, ignore_index=True)

    # 创建一个新的 DataFrame，仅复制前两行
    formatted_df = combined_df.head(2).copy()
    # 应用格式化
    formatted_df['费用'] = formatted_df['费用'].apply(format_first_two_rows, args=('{:.2f}',))
    formatted_df['点击'] = formatted_df['点击'].apply(format_first_two_rows, args=('{:.2f}',))
    formatted_df['GMV'] = formatted_df['GMV'].apply(format_first_two_rows, args=('{:.2f}',))
    formatted_df['ads value'] = formatted_df['ads value'].apply(format_first_two_rows, args=('{:.2f}',))
    formatted_df['神策 ROI'] = formatted_df['神策 ROI'].apply(format_first_two_rows, args=('{:.2f}',))
    formatted_df['ads ROI'] = formatted_df['ads ROI'].apply(format_first_two_rows, args=('{:.2f}',))
    formatted_df['销量'] = formatted_df['销量'].apply(format_first_two_rows, args=('{}',))
    formatted_df['CPC'] = formatted_df['CPC'].apply(format_first_two_rows, args=('{:.2f}',))
    formatted_df['点击率'] = formatted_df['点击率'].apply(format_first_two_rows, args=('{:.2%}',))
    formatted_df['神策转化率'] = formatted_df['神策转化率'].apply(format_first_two_rows, args=('{:.2%}',))
    formatted_df['神策加购率'] = formatted_df['神策加购率'].apply(format_first_two_rows, args=('{:.2%}',))
    formatted_df['支付用户数'] = formatted_df['支付用户数'].apply(format_first_two_rows, args=('{}',))
    formatted_df['UV'] = formatted_df['UV'].apply(format_first_two_rows, args=('{}',))
    formatted_df['加购'] = formatted_df['加购'].apply(format_first_two_rows, args=('{}',))
    formatted_df['展示'] = formatted_df['展示'].apply(format_first_two_rows, args=('{}',))
    formatted_df['ads转化'] = formatted_df['ads转化'].apply(format_first_two_rows, args=('{:.2f}',))
    # 将格式化后的前两行替换回原始 DataFrame
    compare_data_df = combined_df.iloc[2:3].copy()
    compare_data_df[compare_data_df.columns[1:]] = compare_data_df[compare_data_df.columns[1:]].apply(pd.to_numeric,errors='coerce')
    combined_df.update(formatted_df)
    combined_df.update(compare_data_df)
    # combined_df = combined_df.apply(format_comparison, axis=1)
    # combined_df = combined_df.style.apply(colorize_comparison, axis=1)
    return combined_df

def show_df(summary_df):
    st.subheader('SKU数据总览')
    st.dataframe(summary_df
    .style
    .format({
        '费用': '{:.2f}',
        '点击': '{}',
        'GMV': '{:.2f}',
        'ads value': '{:.2f}',
        '神策ROI': '{:.2f}',
        'ads ROI': '{:.2f}',
        'ads转化': '{:.2f}',
        '销量': '{}',
        'CPC': '{:.2f}',
        'CTR': '{:.2%}',
        '神策转化率': '{:.2%}',
        '神策加购率': '{:.2%}',
        '加购': '{}',
        'UV': '{}',
        '支付用户数': '{}'
    })
        , height=500, width=2000
    )

ads_daily = load_and_process_data(ads_url,"0")

seonsor_daily = load_and_process_data(sensor_url,"0")
seonsor_daily = seonsor_daily.rename(columns={'行为时间': 'Date'})

spu_index = load_and_process_data(spu_index_url,"455883801")
process_ads_daily = data_process_daily_ads(ads_daily)
# ads添加SPU
ads_daily_merged_spu_data = get_spu_merged_data(process_ads_daily,spu_index)
# 神策添加SPU
seonsor_daily_merged_spu_data = get_spu_merged_data(seonsor_daily,spu_index)

tab1, tab2 = st.tabs(["SKU汇总查询与对比","SKU日趋势"])
with tab1:
    with st.container(border=True):
        st.subheader('GMV条件输入(广告系列筛选)')
        col1, col2,col3 = st.columns(3)
        # 三个条件范围
        with col1:
            and_tags = st_tags(
            label='“并”条件输入(非完全匹配)',
            )
        with col2:
            or_tags = st_tags(
            label='“或”条件输入(非完全匹配)',
            )
        with col3:
            exclude_tags = st_tags(
            label='排除条件输入(非完全匹配)',
            )
    # 日期范围
    min_date = ads_daily['Date'].min()
    min_date = datetime.strptime(min_date, "%Y-%m-%d")
    max_date = ads_daily['Date'].max()
    max_date = datetime.strptime(max_date, "%Y-%m-%d")
    default_start_date = datetime.today() - timedelta(days=14)
    default_end_date = datetime.today() - timedelta(days=7)

    with st.sidebar:
        selected_range = st.date_input(
            "自选日期范围",
            [default_start_date, default_end_date],
            min_value=min_date,
            max_value=max_date
        )
        compare_selected_range = st.date_input(
            "对比数据日期范围",
            [default_start_date, default_end_date],
            min_value=min_date,
            max_value=max_date
        )
        # 检查会话状态中是否有 'text' 和 'saved_text'，如果没有，初始化它们
        if 'sku_text' not in st.session_state:
            st.session_state['sku_text'] = ""
        if 'sku_saved_text' not in st.session_state:
            st.session_state['sku_saved_text'] = []

        def pass_param():
            # 保存当前文本区域的值到 'saved_text'
            if len(st.session_state['sku_text']) > 0:
                separatedata = st.session_state['sku_text'].split('\n')
                for singedata in separatedata:
                    st.session_state['sku_saved_text'].append(singedata)
            else:
                st.session_state['sku_saved_text'].append(st.session_state['sku_text'])
            # 清空文本区域
            st.session_state['sku_text'] = ""

        def clear_area():
            st.session_state['sku_saved_text'] = []

        # 创建文本区域，其内容绑定到 'text'
        input = st.text_area("批量输入SKU(一个SKU一行)", value=st.session_state['sku_text'], key="sku_text", height=10)

        # 创建一个按钮，当按下时调用 on_upper_clicked 函数
        st.button("确定", on_click=pass_param)

        sku_tags = st_tags(
            label='',
            value=st.session_state['sku_saved_text']  # 使用会话状态中保存的tags
        )
        st.button("清空", on_click=clear_area)

    # 处理神策筛选数据
    # 选择日期范围内的数据
    seonsor_daily_merged_spu_data['Date'] = pd.to_datetime(seonsor_daily_merged_spu_data['Date'])
    ads_daily_merged_spu_data['Date'] = pd.to_datetime(ads_daily_merged_spu_data['Date'])
    # 处理普通选择日期范围内的数据
    seonsor_daily_filtered_date_range_df = seonsor_daily_merged_spu_data[(seonsor_daily_merged_spu_data['Date'] >= pd.to_datetime(selected_range[0])) & (seonsor_daily_merged_spu_data ['Date'] <= pd.to_datetime(selected_range[1]))]
    ads_daily_filtered_date_range_df = ads_daily_merged_spu_data[(ads_daily_merged_spu_data['Date'] >= pd.to_datetime(selected_range[0])) & (ads_daily_merged_spu_data['Date'] <= pd.to_datetime(selected_range[1]))]
    # 处理对比日期范围内的数据
    compare_seonsor_daily_filtered_date_range_df = seonsor_daily_merged_spu_data[(seonsor_daily_merged_spu_data['Date'] >= pd.to_datetime(compare_selected_range[0])) & (seonsor_daily_merged_spu_data['Date'] <= pd.to_datetime(compare_selected_range[1]))]
    compare_ads_daily_filtered_date_range_df = ads_daily_merged_spu_data[(ads_daily_merged_spu_data['Date'] >= pd.to_datetime(compare_selected_range[0])) & (ads_daily_merged_spu_data['Date'] <= pd.to_datetime(compare_selected_range[1]))]

    # 选择筛选条件内的数据
    or_regex = '|'.join(map(str, or_tags))  # 用于“或”筛选
    exclude_regex = '|'.join(map(str, exclude_tags))  # 用于排除
    # 普通日期内的筛选条件
    and_condition = reduce(operator.and_, [seonsor_daily_filtered_date_range_df['Campaign'].str.contains(tag, regex=True, flags=re.IGNORECASE) for tag in and_tags]) if and_tags else True
    or_condition = seonsor_daily_filtered_date_range_df['Campaign'].str.contains(or_regex, regex=True, flags=re.IGNORECASE) if or_tags else True
    exclude_condition = ~seonsor_daily_filtered_date_range_df['Campaign'].str.contains(exclude_regex, regex=True, flags=re.IGNORECASE) if exclude_tags else True
    # 对比日期内的筛选条件
    compare_and_condition = reduce(operator.and_, [compare_seonsor_daily_filtered_date_range_df['Campaign'].str.contains(tag, regex=True, flags=re.IGNORECASE) for tag in and_tags]) if and_tags else True
    compare_or_condition = compare_seonsor_daily_filtered_date_range_df['Campaign'].str.contains(or_regex, regex=True, flags=re.IGNORECASE) if or_tags else True
    compare_exclude_condition = ~compare_seonsor_daily_filtered_date_range_df['Campaign'].str.contains(exclude_regex, regex=True, flags=re.IGNORECASE) if exclude_tags else True

    # 输入了标签才显示数据
    if sku_tags:
       seonsor_condition_sku_filtered_df = seonsor_daily_filtered_date_range_df[seonsor_daily_filtered_date_range_df['SKU'].isin(sku_tags)]
       compare_seonsor_condition_sku_filtered_df = compare_seonsor_daily_filtered_date_range_df[compare_seonsor_daily_filtered_date_range_df['SKU'].isin(sku_tags)]
       if and_tags or or_tags or exclude_tags:
             combined_condition = and_condition & or_condition & exclude_condition
             compare_combined_condition = compare_and_condition & compare_or_condition  & compare_exclude_condition
             seonsor_condition_sku_filtered_df = seonsor_condition_sku_filtered_df[combined_condition]
             compare_seonsor_condition_sku_filtered_df = compare_seonsor_condition_sku_filtered_df[compare_combined_condition]
             sensor_summary_data = create_sensor_summary_data(seonsor_condition_sku_filtered_df)
             summary_df = sensor_and_ads_merged_data(sensor_summary_data,ads_daily_filtered_date_range_df)
             compare_sensor_summary_data = create_sensor_summary_data(compare_seonsor_condition_sku_filtered_df)
             compare_summary_df = sensor_and_ads_merged_data(compare_sensor_summary_data, compare_ads_daily_filtered_date_range_df)
             origin_df = return_combine_all_summary_df(summary_df,selected_range)
             compare_df = return_combine_all_summary_df(compare_summary_df, compare_selected_range)
             all_df = create_compare_summary_df(origin_df,compare_df)

             summary_options = st.multiselect(
                 '选择汇总数据维度',
                 all_df.columns,
                 ['日期范围','费用','GMV']
             )
             all_df = all_df[summary_options]
             all_df = all_df.apply(format_comparison, axis=1)
             all_df = all_df.style.apply(colorize_comparison, axis=1)
             st.subheader('汇总数据对比')
             st.dataframe(all_df)

             options = st.multiselect(
                 '选择总览数据维度',
                 summary_df.columns,
                 ['SPU', 'SKU','Product Type 3','old_or_new','费用', 'GMV', '神策ROI', 'ads ROI']
             )
             show_df(summary_df[options])
       else:
             sensor_summary_data = create_sensor_summary_data(seonsor_condition_sku_filtered_df)
             summary_df = sensor_and_ads_merged_data(sensor_summary_data, ads_daily_filtered_date_range_df)
             compare_sensor_summary_data = create_sensor_summary_data(compare_seonsor_condition_sku_filtered_df)
             compare_summary_df = sensor_and_ads_merged_data(compare_sensor_summary_data,
                                                             compare_ads_daily_filtered_date_range_df)
             origin_df = return_combine_all_summary_df(summary_df,selected_range)
             compare_df = return_combine_all_summary_df(compare_summary_df, compare_selected_range)
             all_df = create_compare_summary_df(origin_df,compare_df)
             summary_options = st.multiselect(
                 '选择数据维度',
                 all_df.columns,
                 ['日期范围','费用','GMV']
             )
             all_df = all_df[summary_options]
             all_df = all_df.apply(format_comparison, axis=1)
             all_df = all_df.style.apply(colorize_comparison, axis=1)
             st.subheader('汇总数据对比')
             st.dataframe(all_df)

             options = st.multiselect(
                 '选择数据维度',
                 summary_df.columns,
                 ['SPU', 'SKU','Product Type 3','old_or_new','费用', 'GMV', '神策ROI', 'ads ROI']
             )
             show_df(summary_df[options])
    else:
          if and_tags or or_tags or exclude_tags:
             combined_condition = and_condition & or_condition & exclude_condition
             seonsor_condition_sku_filtered_df = seonsor_daily_filtered_date_range_df[combined_condition]
             sensor_summary_data = create_sensor_summary_data(seonsor_condition_sku_filtered_df)
             summary_df = sensor_and_ads_merged_data(sensor_summary_data,ads_daily_filtered_date_range_df)
             compare_sensor_summary_data = create_sensor_summary_data(compare_seonsor_daily_filtered_date_range_df)
             compare_summary_df = sensor_and_ads_merged_data(compare_sensor_summary_data,
                                                             compare_ads_daily_filtered_date_range_df)
             origin_df = return_combine_all_summary_df(summary_df, selected_range)
             compare_df = return_combine_all_summary_df(compare_summary_df, compare_selected_range)
             all_df = create_compare_summary_df(origin_df, compare_df)
             summary_options = st.multiselect(
                 '选择数据维度',
                 all_df.columns,
                 ['日期范围', '费用','GMV']
             )
             all_df = all_df[summary_options]
             all_df = all_df.apply(format_comparison, axis=1)
             all_df = all_df.style.apply(colorize_comparison, axis=1)
             st.subheader('汇总数据对比')
             st.dataframe(all_df)

             options = st.multiselect(
                 '选择数据维度',
                 summary_df.columns,
                 ['SPU', 'SKU','Product Type 3','old_or_new','费用', 'GMV', '神策ROI', 'ads ROI']
             )
             show_df(summary_df[options])
          else:
             sensor_summary_data = create_sensor_summary_data(seonsor_daily_filtered_date_range_df)
             summary_df = sensor_and_ads_merged_data(sensor_summary_data,ads_daily_filtered_date_range_df)
             compare_sensor_summary_data = create_sensor_summary_data(compare_seonsor_daily_filtered_date_range_df)
             compare_summary_df = sensor_and_ads_merged_data(compare_sensor_summary_data, compare_ads_daily_filtered_date_range_df)
             origin_df = return_combine_all_summary_df(summary_df, selected_range)
             compare_df = return_combine_all_summary_df(compare_summary_df, compare_selected_range)
             all_df = create_compare_summary_df(origin_df, compare_df)
             summary_options = st.multiselect(
                 '选择数据维度',
                 all_df.columns,
                 ['日期范围', '费用','GMV']
             )
             all_df = all_df[summary_options]
             all_df = all_df.apply(format_comparison, axis=1)
             all_df = all_df.style.apply(colorize_comparison, axis=1)
             st.subheader('汇总数据对比')
             st.dataframe(all_df)

             options = st.multiselect(
                 '选择数据维度',
                 summary_df.columns,
                 ['SPU', 'SKU','Product Type 3','old_or_new','费用', 'GMV', '神策ROI', 'ads ROI']
             )
             show_df(summary_df[options])
with tab2:
    if sku_tags:
     ads_condition_sku_filtered_df = ads_daily_filtered_date_range_df[ads_daily_filtered_date_range_df['SKU'].isin(sku_tags)]
     ads_condition_sku_filtered_df['Date'] = pd.to_datetime(ads_condition_sku_filtered_df['Date']).dt.strftime('%Y-%m-%d')
     ads_pivot_df = ads_condition_sku_filtered_df.pivot(index=['SKU'], columns='Date', values='cost')
     st.dataframe(ads_pivot_df)
