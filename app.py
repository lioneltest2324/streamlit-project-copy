import streamlit as st
from st_pages import Page, show_pages
from streamlit_gsheets import GSheetsConnection
import pandas as pd
st.set_page_config(layout="wide",page_title='SPU数据跟踪',page_icon = 'favicon.png')
show_pages(
    [
        Page("app.py", "可编辑SPU登记表格"),
        Page("spu_daily_view.py","SPU日维度数据跟踪"),
        Page("sku_monthly_view.py", "SPU所属SKU-2023年度数据查看"),
        Page("sku_data.py","SKU数据查看"),
        Page("campaign_sku.py","购物广告内部结构查看")
    ]
)

st.markdown("<h1 style='text-align: center;'>SPU数据跟踪</h1>", unsafe_allow_html=True)
with st.container():
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        expander = st.expander("SPU测试纪要(SKU转SPU测试计划)")
        expander.write("""
        纯老品SKU转SPU测试计划:\n
        测试目的：\n
        减少同一个SPU下效果差产品的重复花费，进行SKU投放剔除\n
        SPU测试前提：\n
        一、一年花费过低的SPU无法做出效果判断，因此暂不做SKU剔除(月均花费>50)\n
        二、暂不对头部产生GMV的SPU操作(月均GMV<5000)\n
        三、只对中尾部的效果差的SPU操作(1-11月ROI<2)\n
        四、由于是缩减SKU数的测试，因此SPU下的SKU需要≥2(SKU数量≥2)\n
        五、测试初期先对头部三级类目进行测试，进而快速回收效果(三级类目：daybed，dining table，tv stand)\n
        六、剔除户外一级类目，这部分交给婷婷\n
        SKU筛选逻辑：\n
        一、为了排除偶然出单带来的高ROI导致数据判断错误，需要对SKU花费做判断(单SKU月均花费>SPU/SKU个数的月均花费)\n
        二、排除下架产品\n
        三、数据优先级(加入引流款判断)：\n
        1.神策ROI\n
        2.ads-ROI\n
        3.神策-GMV下半年趋势\n
        4.ads-GMV下半年趋势\n
        5.CTR\n
        """)
url = 'https://docs.google.com/spreadsheets/d/1ZqP2C9a7NG5ksJf_5RyIbIQhVqmFygr0FbsTrJPSqLw/edit#gid=0'
st.subheader('新老品登记编辑')
conn = st.connection("gsheets", type=GSheetsConnection)
edite_source = conn.read(spreadsheet=url,worksheet='0',ttl="1m")
daily_df = pd.DataFrame(edite_source)
daily_df = daily_df.dropna(subset=['SPU'], how='all')
edit_df = st.dataframe(daily_df,
                         column_config={
                             "图片": st.column_config.ImageColumn(
                                 "图片",
                                 width="small"
                             )},
                        width=1400,height=400,hide_index=True)
