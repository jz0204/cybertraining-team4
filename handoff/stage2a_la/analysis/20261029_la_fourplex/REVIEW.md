# 31个四户候选逐栋复核

复核62个轮廓：28栋主楼加入多户训练，28个车库/车棚标记为附属结构并排除用途训练；其余3个地块的6个轮廓保留unknown。判断结合官方记录和真实火前近景/上下文影像，由AI辅助完成，未经过独立人工复核，不能称为金标准。

纳入的28个地块均登记4户、1个建筑记录、8卧8卫、主建筑面积3868平方英尺。前排主楼屋顶195–254平方米，楼面/屋顶比1.41–1.84；后排结构67–83平方米，报告高度明显较低，并与停车通道相连。这些证据共同支持主楼与车库/车棚的区分，不表示影像能直接数出4户。28例来自同一街区，外观高度相关。

原3332测试记录、438验证记录、输入路径及标签完整保留；新增56个CNN输入文件均完整保留目标屋顶，未与旧2km测试排除区、验证上下文或验证网格重叠。附属建筑用途不强行归入三类CNN。

来源：[2024 assessor表](https://services.arcgis.com/RmCCgQtiZLDCtblq/ArcGIS/rest/services/Parcel_Data_2021_Table/FeatureServer/0)、[2014建筑轮廓](https://dpw.gis.lacounty.gov/dpw/rest/services/FEMAFloodzoneAdopted/MapServer/15)、[2020建筑轮廓](https://public.gis.lacounty.gov/public/rest/services/LACounty_Dynamic/LARIAC_Buildings_2020/MapServer/0)。原始响应、查询参数及哈希保存在外置盘raw目录；逐楼影像及标记保留在images目录。

|序号|地块|A较大轮廓|B较小轮廓|影像页|
|---|---|---|---|---|
|1|2645013003|unknown，排除|unknown，排除|[近景和上下文](/Users/zhou/Documents/GitHub/cybertraining-team4/handoff/stage2a_la/analysis/20261029_la_fourplex/contact_sheets/sheet_01.jpg)|
|2|2617003052|unknown，排除|unknown，排除|[近景和上下文](/Users/zhou/Documents/GitHub/cybertraining-team4/handoff/stage2a_la/analysis/20261029_la_fourplex/contact_sheets/sheet_01.jpg)|
|3|2535006002|unknown，排除|unknown，排除|[近景和上下文](/Users/zhou/Documents/GitHub/cybertraining-team4/handoff/stage2a_la/analysis/20261029_la_fourplex/contact_sheets/sheet_01.jpg)|
|4|2784022001|纳入：多户主楼|附属车库/车棚，排除|[近景和上下文](/Users/zhou/Documents/GitHub/cybertraining-team4/handoff/stage2a_la/analysis/20261029_la_fourplex/contact_sheets/sheet_01.jpg)|
|5|2784022006|纳入：多户主楼|附属车库/车棚，排除|[近景和上下文](/Users/zhou/Documents/GitHub/cybertraining-team4/handoff/stage2a_la/analysis/20261029_la_fourplex/contact_sheets/sheet_02.jpg)|
|6|2784005056|纳入：多户主楼|附属车库/车棚，排除|[近景和上下文](/Users/zhou/Documents/GitHub/cybertraining-team4/handoff/stage2a_la/analysis/20261029_la_fourplex/contact_sheets/sheet_02.jpg)|
|7|2784022010|纳入：多户主楼|附属车库/车棚，排除|[近景和上下文](/Users/zhou/Documents/GitHub/cybertraining-team4/handoff/stage2a_la/analysis/20261029_la_fourplex/contact_sheets/sheet_02.jpg)|
|8|2784022008|纳入：多户主楼|附属车库/车棚，排除|[近景和上下文](/Users/zhou/Documents/GitHub/cybertraining-team4/handoff/stage2a_la/analysis/20261029_la_fourplex/contact_sheets/sheet_02.jpg)|
|9|2784022018|纳入：多户主楼|附属车库/车棚，排除|[近景和上下文](/Users/zhou/Documents/GitHub/cybertraining-team4/handoff/stage2a_la/analysis/20261029_la_fourplex/contact_sheets/sheet_03.jpg)|
|10|2784005051|纳入：多户主楼|附属车库/车棚，排除|[近景和上下文](/Users/zhou/Documents/GitHub/cybertraining-team4/handoff/stage2a_la/analysis/20261029_la_fourplex/contact_sheets/sheet_03.jpg)|
|11|2784005055|纳入：多户主楼|附属车库/车棚，排除|[近景和上下文](/Users/zhou/Documents/GitHub/cybertraining-team4/handoff/stage2a_la/analysis/20261029_la_fourplex/contact_sheets/sheet_03.jpg)|
|12|2784022007|纳入：多户主楼|附属车库/车棚，排除|[近景和上下文](/Users/zhou/Documents/GitHub/cybertraining-team4/handoff/stage2a_la/analysis/20261029_la_fourplex/contact_sheets/sheet_03.jpg)|
|13|2784022017|纳入：多户主楼|附属车库/车棚，排除|[近景和上下文](/Users/zhou/Documents/GitHub/cybertraining-team4/handoff/stage2a_la/analysis/20261029_la_fourplex/contact_sheets/sheet_04.jpg)|
|14|2784022002|纳入：多户主楼|附属车库/车棚，排除|[近景和上下文](/Users/zhou/Documents/GitHub/cybertraining-team4/handoff/stage2a_la/analysis/20261029_la_fourplex/contact_sheets/sheet_04.jpg)|
|15|2784022005|纳入：多户主楼|附属车库/车棚，排除|[近景和上下文](/Users/zhou/Documents/GitHub/cybertraining-team4/handoff/stage2a_la/analysis/20261029_la_fourplex/contact_sheets/sheet_04.jpg)|
|16|2784022013|纳入：多户主楼|附属车库/车棚，排除|[近景和上下文](/Users/zhou/Documents/GitHub/cybertraining-team4/handoff/stage2a_la/analysis/20261029_la_fourplex/contact_sheets/sheet_04.jpg)|
|17|2784022009|纳入：多户主楼|附属车库/车棚，排除|[近景和上下文](/Users/zhou/Documents/GitHub/cybertraining-team4/handoff/stage2a_la/analysis/20261029_la_fourplex/contact_sheets/sheet_05.jpg)|
|18|2784022020|纳入：多户主楼|附属车库/车棚，排除|[近景和上下文](/Users/zhou/Documents/GitHub/cybertraining-team4/handoff/stage2a_la/analysis/20261029_la_fourplex/contact_sheets/sheet_05.jpg)|
|19|2784022012|纳入：多户主楼|附属车库/车棚，排除|[近景和上下文](/Users/zhou/Documents/GitHub/cybertraining-team4/handoff/stage2a_la/analysis/20261029_la_fourplex/contact_sheets/sheet_05.jpg)|
|20|2784022019|纳入：多户主楼|附属车库/车棚，排除|[近景和上下文](/Users/zhou/Documents/GitHub/cybertraining-team4/handoff/stage2a_la/analysis/20261029_la_fourplex/contact_sheets/sheet_05.jpg)|
|21|2784022014|纳入：多户主楼|附属车库/车棚，排除|[近景和上下文](/Users/zhou/Documents/GitHub/cybertraining-team4/handoff/stage2a_la/analysis/20261029_la_fourplex/contact_sheets/sheet_06.jpg)|
|22|2784005052|纳入：多户主楼|附属车库/车棚，排除|[近景和上下文](/Users/zhou/Documents/GitHub/cybertraining-team4/handoff/stage2a_la/analysis/20261029_la_fourplex/contact_sheets/sheet_06.jpg)|
|23|2784022011|纳入：多户主楼|附属车库/车棚，排除|[近景和上下文](/Users/zhou/Documents/GitHub/cybertraining-team4/handoff/stage2a_la/analysis/20261029_la_fourplex/contact_sheets/sheet_06.jpg)|
|24|2784005050|纳入：多户主楼|附属车库/车棚，排除|[近景和上下文](/Users/zhou/Documents/GitHub/cybertraining-team4/handoff/stage2a_la/analysis/20261029_la_fourplex/contact_sheets/sheet_06.jpg)|
|25|2784005054|纳入：多户主楼|附属车库/车棚，排除|[近景和上下文](/Users/zhou/Documents/GitHub/cybertraining-team4/handoff/stage2a_la/analysis/20261029_la_fourplex/contact_sheets/sheet_07.jpg)|
|26|2784022016|纳入：多户主楼|附属车库/车棚，排除|[近景和上下文](/Users/zhou/Documents/GitHub/cybertraining-team4/handoff/stage2a_la/analysis/20261029_la_fourplex/contact_sheets/sheet_07.jpg)|
|27|2784022015|纳入：多户主楼|附属车库/车棚，排除|[近景和上下文](/Users/zhou/Documents/GitHub/cybertraining-team4/handoff/stage2a_la/analysis/20261029_la_fourplex/contact_sheets/sheet_07.jpg)|
|28|2784005049|纳入：多户主楼|附属车库/车棚，排除|[近景和上下文](/Users/zhou/Documents/GitHub/cybertraining-team4/handoff/stage2a_la/analysis/20261029_la_fourplex/contact_sheets/sheet_07.jpg)|
|29|2784005048|纳入：多户主楼|附属车库/车棚，排除|[近景和上下文](/Users/zhou/Documents/GitHub/cybertraining-team4/handoff/stage2a_la/analysis/20261029_la_fourplex/contact_sheets/sheet_08.jpg)|
|30|2784005057|纳入：多户主楼|附属车库/车棚，排除|[近景和上下文](/Users/zhou/Documents/GitHub/cybertraining-team4/handoff/stage2a_la/analysis/20261029_la_fourplex/contact_sheets/sheet_08.jpg)|
|31|2784005058|纳入：多户主楼|附属车库/车棚，排除|[近景和上下文](/Users/zhou/Documents/GitHub/cybertraining-team4/handoff/stage2a_la/analysis/20261029_la_fourplex/contact_sheets/sheet_08.jpg)|

第4例B轮廓缺少2020同ID记录，但当前影像显示后排低矮停车屋顶。它只保留附属标记，不加入训练；其A主楼通过2014/2020稳定对应。前三例的具体疑点和所有逐栋判断见BUILDING_REVIEW.csv/REVIEW_SOURCE.json。
