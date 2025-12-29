import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import re
# import llm_caller
from pathlib import Path

# 根据你的282个token人工归纳出30个核心类别
PARKING_MAP = {
    # 1. 核心车库类型（权重最高）
    'attached_garage': ['attached', 'attached garage'],
    'detached_garage': ['detached', 'detached garage'],
    'carport': ['carport', 'car port', 'covered carport'],
    'garage': ['garage', 'garage door', 'garage door opener'],  # 通用garage
    
    # 2. 车库属性
    'shed': ['shed', 'storage'],
    'workshop': ['workshop', 'workshop in garage', 'mechanical lift'],
    'heated_garage': ['heated garage', 'insulated'],
    
    # 3. 车道类型
    'driveway': ['driveway', 'driveway brick', 'driveway combination', 'driveway level'],
    'circular_driveway': ['circular driveway', 'porte-cochere'],
    'shared_driveway': ['shared driveway', 'rotational'],
    
    # 4. 地面材质
    'paved': ['paved', 'asphalt', 'concrete'],
    'gravel': ['gravel'],
    'unpaved': ['unpaved'],
    
    # 5. 停车管理方式
    'assigned': ['assigned', 'space per unit', 'unassigned'],
    'valet': ['valet'],
    'gated': ['gated', 'controlled entrance', 'secured'],
    
    # 6. 特殊停车
    'rv_parking': ['rv', 'rv garage', 'rv access', 'rv storage', 'rv gated', 'boat'],
    'underground': ['underground', 'below ground parking', 'below building parking'],
    'open_parking': ['open', 'outdoor'],
    'street_parking': ['on street', 'on-street', 'alley access', 'alley'],
    
    # 7. 容量描述
    'multiple_cars': ['4-7 cars', 'four +', 'four or more', 'two', 'three', 'one'],
    'tandem': ['tandem', 'tandem covered', 'tandem uncovered'],
    
    # 8. 无/限制情况
    'no_parking': ['none', 'no garage', 'no driveway', 'no parking on site'],
    'limited': ['size limited', 'see remarks', 'buyer to verify', 'permit required'],
    
    # 9. 现代设施
    'ev_charging': ['electric vehicle charging'],
    
    # 10. 其他
    'bonus_area': ['bonus area inside', 'built-in storage'],
}

LAUNDRY_MAP = {
    # 1. 核心位置（13类）
    'basement': ['basement', 'lower level', 'downstairs'],
    'garage': ['garage', 'laundry area in garage', 'laundry in garage'],
    'kitchen': ['kitchen', 'laundry area in kitchen'],
    'closet': ['closet', 'laundry closet', 'laundry closet stacked'],
    'utility_room': ['utility room', 'in utility room', 'laundry room'],
    'bathroom': ['bathroom', 'bath'],
    'porch': ['porch', 'on porch'],
    'hallway': ['hall', 'hallway'],
    'bedroom': ['bedroom'],
    'carport': ['in carport', 'carport'],
    'outside': ['outside', 'outdoor', 'on-site', 'on site'],
    'inside': ['inside', 'inside area', 'inside room', 'in house'],
    'main_level': ['main level', 'ground floor', 'upper level', 'upper floor'],
    
    # 2. 设备类型（6类）
    'washer_dryer': ['washer_dryer', 'washer dryer', 'washer/dryer'],
    'washer_only': ['washer', 'washer included'],
    'dryer_only': ['dryer', 'dryer included'],
    'washer_dryer_hookup': ['washer_dryer_hookup', 'washer hookup', 'dryer hookup'],
    'stacked': ['stacked', 'stacked space', 'washer/dryer stacked'],
    'hookup_only': ['hookup', 'hookups', 'hookups only', 'no hookups'],
    
    # 3. 能源类型（4类）
    'electric': ['electric', '110v', '220v', 'electric dryer hookup'],
    'gas': ['gas', 'gas dryer hookup', 'propane', 'propane dryer hookup'],
    'gas_electric': ['gas & electric dryer hookup', 'gas/electric dryer hookup'],
    'propane': ['propane', 'propane dryer hookup'],
    
    # 4. 设施状态（6类）
    'coin_op': ['coin op', 'coin operated'],
    'common_area': ['common area', 'common', 'community', 'community facility'],
    'assigned': ['assigned', 'space per unit', 'unassigned'],
    'storage': ['storage', 'cabinets', 'shelves', 'with storage', 'built-in storage'],
    'sink': ['sink', 'utility sink', 'tub/sink', 'with a sink'],
    'chute': ['chute', 'laundry chute'],
    
    # 5. 无/限制情况（4类）
    'none': ['none', 'no laundry fac in unit', 'not in unit'],
    'no_hookups': ['no hookups'],
    'varies': ['varies by unit', 'unknown'],
    'other': ['other', 'other-rmks', 'see remarks', 'other-attch'],
    
    # 6. 特殊配置（2类）
    'separate_room': ['separate room', 'separate', 'individual room'],
    'stackable_unit': ['stackable', 'stacked only'],
}

APPLIANCES_MAP = {
    # 第一级：核心设备（10类）
    'refrigerator': ['refrigerator', 'wine refrigerator', 'freezer', 'ice maker', 'water line', 'plumbed ice maker'],
    'dishwasher': ['dishwasher', 'portable dishwasher'],
    'microwave': ['microwave'],
    'oven_range': ['oven', 'range', 'cooktop', 'stove', '6 burner', 'double oven', 'convection oven', 'self cleaning oven', 'warming drawer', 'range hood'],
    'washer': ['washer'],
    'dryer': ['dryer'],
    'washer_dryer_combo': ['washerdryer', 'washer dryer stacked'],
    'trash_compactor': ['trash compactor'],
    'disposal': ['disposal', 'garbage disposal'],
    'bbq_grill': ['barbecue', 'grill builtin', 'indoor grill'],
    
    # 第二级：能源系统（8类）
    'water_heater_electric': ['electric water heater', '220v', '110v', 'instant hot water', 'hot water circulator'],
    'water_heater_gas': ['gas water heater', 'gas plumbed'],
    'water_heater_propane': ['propane water heater'],
    'water_heater_tankless': ['tankless water heater'],
    'water_heater_solar': ['solar water heater', 'solar hot water'],
    'water_heater_high_eff': ['high efficiency water heater'],
    'water_heater_insulated': ['insulated water heater'],
    'water_heater_coal': ['coal water heater'],
    
    # 第三级：水处理设备（2类）
    'water_softener': ['water softener'],
    'water_purifier': ['water purifier', 'water filter system'],
    
    # 第四级：烹饪能源类型（4类）
    'cooking_electric': ['electric cooking', 'electric range', 'electric cooktop', 'electric oven', 'all electric'],
    'cooking_gas': ['gas cooking', 'gas range', 'gas cooktop', 'gas oven'],
    'cooking_propane': ['propane cooking', 'propane range', 'propane cooktop', 'propane oven'],
    'cooking_gas_electric': ['gas_electric'],
    
    # 第五级：特殊认证与功能（2类）
    'energy_star': ['energy star'],
    'exhaust_fan': ['exhaust fan', 'recirculated exhaust fan', 'vented exhaust fan'],
    
    # 第六级：交易状态（4类）
    'negotiable': ['refrig negotiable', 'stove is tenants own'],
    'none': ['none'],
    'unknown': ['unknown', 'see remarks', 'missing', 'submit'],
    'other': ['other appl available', 'space for frzr/refr'],
}

COOLING_MAP = {
    # 第一级：核心空调类型（10类）
    'central_ac': ['central_ac', 'zoned_ac', 'dual', 'mixed_central', 'central_ducted'],
    'evaporative_cooler': ['evaporative_central', 'evaporative_hallway', 'evaporative_window', 'evaporative_cooler'],
    'window_ac': ['window_ac', 'wall_ac'],
    'mini_split': ['mini_split', 'heat_pump'],  # 高端分体式
    'none': ['none'],
    
    # 第二级：辅助通风（3类）
    'ceiling_fan': ['ceiling_fan', 'ceiling fans pre-wired'],
    'attic_fan': ['attic_fan'],
    'whole_house_fan': ['whole_house_fan'],
    
    # 第三级：能源类型（3类）
    'electric_only': ['electric_only', 'electric'],  # 全电系统
    'gas': ['gas'],
    'solar': ['solar'],
    
    # 第四级：效率认证（3类）
    'energy_star': ['energy_star'],
    'seer_16_plus': ['seer_16_plus'],
    'seer_13_15': ['seer_13_15'],
    
    # 第五级：高级功能（2类）
    'high_efficiency': ['high_efficiency'],
    'humidity_control': ['humidity_control'],
    
    # 第六级：未知/其他（1类）
    'unknown': ['unknown', 'geothermal', 'other', 'recent_replacement'],
}

HEATING_MAP = {
    # 第一级：核心供暖系统（12类，权重最高）
    'forced_air': ['forced_air', 'forced_air_gas', 'forced_air_electric'],
    'forced_air_gas': ['forced_air_gas'],
    'forced_air_electric': ['forced_air_electric'],
    'heat_pump': ['heat_pump'],
    'mini_split': ['mini_split'],
    'radiant_floor': ['radiant_floor'],
    'hydronic': ['hydronic'],
    'radiant': ['radiant', 'perimeter_radiant'],
    'baseboard': ['baseboard'],
    'steam': ['steam'],
    'gravity': ['gravity'],
    'wall_furnace': ['wall_furnace'],
    'floor_furnace': ['floor_furnace'],
    
    # 第二级：复合/混合系统（1类）
    'mixed_system': ['mixed_system'],  # Combination/双系统
    
    # 第三级：辅助供暖（3类）
    'fireplace': ['fireplace'],
    'pellet_stove': ['pellet_stove', 'wood_pellet'],
    'stove': ['stove'],
    'space_heater': ['space_heater'],
    
    # 第四级：能源类型（8类）
    'gas': ['gas'],
    'electric': ['electric'],
    'solar': ['solar'],
    'geothermal': ['geothermal'],
    'wood': ['wood'],
    'propane': ['propane'],
    'oil': ['oil'],
    'kerosene': ['kerosene'],
    'coal': ['coal'],
    
    # 第五级：效率与认证（3类）
    'energy_star': ['energy_star'],
    'high_efficiency': ['high_efficiency'],
    
    # 第六级：分布控制（2类）
    'zoned': ['zoned'],
    'smart_vent': ['smart_vent'],
    'humidity_control': ['humidity_control'],
    
    # 第七级：状态（2类）
    'none': ['none'],
    'unknown': ['unknown'],
}

FLOORING_MAP = {
    # 第一级：全屋高端木材（溢价标记）
    'hardwood_throughout': ['hardwood_throughout'],  # 全屋硬木，价值+15%
    
    # 第二级：优质木材（独立类别）
    'hardwood': ['hardwood'],
    'engineered_wood': ['engineered_wood'],
    'reclaimed_wood': ['reclaimed_wood'],
    'bamboo': ['bamboo'],
    'softwood': ['softwood'],
    
    # 第三级：天然石材/瓷砖（高价值）
    'stone': ['stone', 'granite', 'marble'],
    'tile': ['tile'],
    'travertine': ['travertine'],
    
    # 第四级：复合地板（中端）
    'laminate': ['laminate', 'wood_laminate', 'wood_like'],
    
    # 第五级：经济地板（基础）
    'vinyl': ['vinyl'],
    'cork': ['cork'],
    'carpet': ['carpet'],
    
    # 第六级：硬质地面（特殊用途）
    'concrete': ['concrete', 'stamped_concrete'],
    'brick': ['brick', 'adobe'],
    'pavers': ['pavers'],
    'parquet': ['parquet'],
    
    # 第七级：特殊状态（负向价值）
    'partial_carpet': ['partial_carpet'],  # 部分地毯=部分旧房
    'mixed_flooring': ['mixed', 'combination'],  # 混合材料=复杂
    'unfinished': ['unfinished', 'no flooring'],
    'new_flooring': ['new_flooring'],  # 新装修，价值+
    
    # 第八级：未知/其他
    'unknown': ['unknown', 'varies', 'painted'],
}

def drop_samples(features):
    # 删除缺失值的比例大于30%的行，即仅保留缺失值小于30%的行
    features = features[features.isna().mean(axis=1)<0.3]
    # 删除税收评估价格、税额缺失的行，因为这两项对房价影响较大，不好填充
    features = features.dropna(subset=['Tax assessed value', 'Annual tax amount'])
    # 删除数值太大或太小，不真实的行
    features = features[~( # 删掉满足这些条件的行，即仅不保留满足这些条件的行，~用于取反（False变成True）
        (features['Total interior livable area'] >= 1e6) |
        (features['Total interior livable area'] <= 60) |
        (features['Lot'] <= 300) | # 少了1W
        (features['Lot'] >= 4e6) |
        (features['Bedrooms'] >= 20) |
        (features['Garage spaces'] >= 500) |
        (features['Garage spaces'] < 0) |
        (features['Listed Price'] < 500) |
        (features['Last Sold Price'] < 500)
    )]
    return features

def roomstr_to_num(x):
    """将字符串类型的房间数特征转换为数值类型"""
    if pd.isna(x):  # 统一处理 None, np.nan, pd.NA
        return np.nan
    if x.isdigit():
        return int(x)
    return len(x.split(','))

def clean_parking_token(text):
    """暴力清洗：删除噪音，统一格式"""
    if pd.isna(text):
        return ''
    
    # 1. 转为小写
    text = str(text).lower()
    
    # 2. 删除所有测量数据和编号 (2)10x12, Cpt #1, 24'+ Deep
    text = re.sub(r'\(\d+\)\d+x\d+|\d+\'+|deep|wide|cpt\s*#\d+|gar\s*#\d+|unc\s*#\d+', '', text)

    # 3. 删除语法词和停用词
    text = re.sub(r'\b(is|converted|from|street|on|site|in|the|and|or)\b', '', text)
    
    # 4. 删除连字符和多余空格
    text = text.replace('-', ' ').replace('/', ' ')
    text = ' '.join(text.split())  # 去重空格
    
    # 5. 删除纯数字token
    text = re.sub(r'\b\d+\b', '', text)
    
    # 6. 删除单字符
    text = re.sub(r'\b\w\b', '', text)
    
    return text.strip()

def clean_laundry_token(text):
    """Laundry专用清洗：处理电压、位置、设备变体"""
    if pd.isna(text):
        return ''
    
    text = str(text).lower()
    
    # 1. 删除编号、尺寸、噪音词
    text = re.sub(r'\b\d+\b|cpt\s*#\d+|gar\s*#\d+|unc\s*#\d+', '', text)
    text = re.sub(r'\(\d+\)', '', text)  # 删除(110V)(220V)括号
    
    # 2. 统一电压描述：220V Elect → 220v, 220 Volt Outlet → 220v
    text = re.sub(r'220\s*volt\s*outlet|220v\s*elect|electricity\s*hookup\s*\(220v\)', '220v', text)
    text = re.sub(r'electricity\s*hookup\s*\(110v\)', '110v', text)
    text = re.sub(r'electric\s*hook\s*-*\s*up|electric\s*hookup', 'electric', text)
    
    # 3. 统一Hookup变体
    text = re.sub(r'hook\s*-*\s*up', 'hookup', text)
    text = re.sub(r'hookups\s*only', 'hookup', text)
    
    # 4. 统一位置描述：In Basement → basement
    text = re.sub(r'\bin\b\s*(basement|garage|kitchen|closet|unit|house|utility room|carport)', r'\1', text)
    text = re.sub(r'\bon\b\s*(lower level|upper level|porch)', r'\1', text)
    
    # 5. 统一Washer/Dryer变体
    text = re.sub(r'washer\s*/\s*dryer\s*hookups?', 'washer_dryer_hookup', text)
    text = re.sub(r'washer\s*/\s*dryer\s*stacked\s*incl', 'washer_dryer_stacked', text)
    text = re.sub(r'washer\s*/\s*dryer', 'washer_dryer', text)
    text = re.sub(r'washer\s*hookups?', 'washer', text)
    text = re.sub(r'dryer\s*hookups?', 'dryer', text)
    
    # 6. 统一Stacked变体
    text = re.sub(r'stacked\s*only|stackable', 'stacked', text)
    
    # 7. 删除停用词
    text = re.sub(r'\b(and|or|the|a|an|in|on|with|by|is|see|remarks|other|spec)\b', '', text)
    
    # 8. 清理多余空格
    text = ' '.join(text.split())
    
    return text.strip()

def clean_appliances_token(text):
    """Appliances专用清洗：统一设备、能源、型号描述"""
    if pd.isna(text):
        return ''
    
    text = str(text).lower()
    
    # 1. 删除数量前缀（如"1 Refrigerator"→"refrigerator"）
    text = re.sub(r'^\d+\s+', '', text)
    
    # 2. 统一标准词
    replacements = {
        'energy star qualified': 'energy_star',
        'free standing': 'freestanding',
        'built in': 'builtin',
        'gas & electric': 'gas_electric',
        'gas/elec': 'gas_electric',
        'washer/dryer': 'washerdryer',
        'range/oven': 'range_oven',
        'plumbed for': 'plumbed',
        'all electric': 'electric',
        'see remarks': 'unknown',
        'other appl available': 'other',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # 3. 删除特殊字符，保留关键词
    text = re.sub(r'[^\w\s]', ' ', text)
    text = ' '.join(text.split())
    
    return text

def clean_appliances_token_regex(text):
    """Appliances included - 正则版清洗"""
    if pd.isna(text):
        return ''
    
    text = str(text).lower()
    
    # 1. 删除数量前缀和纯数字编号
    text = re.sub(r'^\d+\s+|\b\d+\b', '', text)
    
    # 2. 统一设备类型（按优先级）
    patterns = [
        (r'free\s*standing', 'freestanding'),
        (r'washer\s*/\s*dryer\s*hookups?', 'washerdryerhookup'),
        (r'washer\s*/\s*dryer\s*stacked\s*inch', 'washerdryerstacked'),
        (r'washer\s*/\s*dryer', 'washerdryer'),
        (r'washer\s*hookups?', 'washer'),
        (r'dryer\s*hookups?', 'dryer'),
        (r'energy\s+star\s+qualified\s+equipment', 'energystar'),
        (r'plumbed\s+for\s+ice\s+maker', 'icemakerr plumbed'),
        (r'energy\s+star\s+qualified', 'energystar'),
        (r'high\s+efficiency', 'highefficiency'),
        (r'outdoor\s+grill', 'grill'),
        (r'indoor\s+grill', 'grill'),
        (r'portable\s+dishwasher', 'dishwasher'),
        (r'free\s+zer', 'freezer'),
    ]
    for pattern, repl in patterns:
        text = re.sub(pattern, repl, text)
    
    # 3. 统一能源系统
    energy_patterns = [
        (r'electric\s+water\s+heater|electric\s+heater', 'electricwaterheater'),
        (r'gas\s+water\s+heater|gas\s+heater', 'gaswaterheater'),
        (r'propane\s+water\s+heater|propane\s+heater', 'propanewaterheater'),
        (r'tankless\s+water\s+heater', 'tanklesswaterheater'),
        (r'solar\s+water\s+heater|solar\s+hot\s+water', 'solarwaterheater'),
        (r'insulated\s+water\s+heater', 'insulatedwaterheater'),
        (r'coal\s+water\s+heater', 'coalwaterheater'),
        (r'electric\s+cooking|all\s+electric', 'electriccooking'),
        (r'gas\s+cooking', 'gascooking'),
        (r'propane\s+cooking', 'propanecooking'),
        (r'gas\s+&\s+electric\s+range|gas/electric', 'gaselectric'),
    ]
    for pattern, repl in energy_patterns:
        text = re.sub(pattern, repl, text)
    
    # 4. 特殊标记
    text = re.sub(r'trash\s+compactors?', 'trashcompactor', text)
    text = re.sub(r'water\s+softeners?', 'watersoftener', text)
    text = re.sub(r'water\s+purifiers?', 'waterpurifier', text)
    
    # 5. 删除停用词和特殊字符
    text = re.sub(r'\b(and|or|the|a|an|in|on|with|by|is|to|for|of)\b', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = ' '.join(text.split())
    
    return text.strip()


def clean_cooling_token(text):
    """Cooling专用清洗：统一AC/Evap/Split等核心词"""
    if pd.isna(text):
        return ''
    
    text = str(text).lower()
    
    # 1. 删除括号、连字符、斜杠
    text = text.replace('/', ' ').replace('-', ' ').replace('(', '').replace(')', '')
    
    # 2. 统一核心词
    replacements = {
        'ac zoned': 'zoned_ac',
        'ac central': 'central_ac',
        'ac wall': 'wall_ac',
        'central air': 'central_ac',
        'central air/refrig': 'central_ac',
        'central air/evap': 'mixed_central',
        'refridge wall/window': 'window_ac',
        'refrigerator wall/window': 'window_ac',
        'room air': 'window_ac',
        'wall/window unit': 'window_ac',
        'window unit': 'window_ac',
        'wall unit': 'window_ac',
        'evap central': 'evaporative_central',
        'evap hallway': 'evaporative_hallway',
        'evap wall/window': 'evaporative_window',
        'evap cooler': 'evaporative_cooler',
        'swamp cooler': 'evaporative_cooler',
        'evaporative cooling': 'evaporative_cooler',
        'mini split ac': 'mini_split',
        'mini-split': 'mini_split',
        'split system': 'mini_split',
        'heat pump': 'heat_pump',
        'whole house fan': 'whole_house_fan',
        'ceiling fan': 'ceiling_fan',
        'attic fan': 'attic_fan',
        'no air conditioning': 'none',
        'seer rated 13-15': 'seer_13_15',
        'seer rated 16+': 'seer_16_plus',
        'energy star qualified equipment': 'energy_star',
        'high efficiency': 'high_efficiency',
        'all electric': 'electric_only',
        'dual cooling': 'dual',
        'master cooler': 'evaporative_cooler',
        'buyer to verify': 'unknown',
        'see remarks': 'unknown',
        'new construction option': 'unknown',
        'unit replaced 5 yrs': 'recent_replacement',
        'has ducting hvac': 'central_ducted',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # 3. 删除停用词
    text = re.sub(r'\b(and|or|the|a|an|in|on|with|by|is|to|for|unit|s)\b', '', text)
    
    # 4. 去重空格
    text = ' '.join(text.split())
    
    return text.strip()


def clean_heating_token(text):
    """Heating专用清洗：统一系统类型、能源、位置描述"""
    if pd.isna(text):
        return ''
    
    text = str(text).lower()
    
    # 1. 删除编号、连字符、斜杠
    text = text.replace('-', ' ').replace('/', ' ').replace('(', '').replace(')', '')
    
    # 2. 核心标准化（顺序不能乱）
    replacements = {
        'cfah': 'forced_air',  # CFAH是强制通风缩写
        'forced air - gas': 'forced_air_gas',
        'forced air - elec': 'forced_air_electric',
        'central heat/gas': 'forced_air_gas',
        'central heat/elec': 'forced_air_electric',
        'central heat': 'forced_air',  # Central Heat默认=强制通风
        'heat pump-air': 'heat_pump',
        'heat pump': 'heat_pump',
        'wall unit': 'wall_furnace',
        'wall furnace': 'wall_furnace',
        'floor furnace': 'floor_furnace',
        'radiant floor': 'radiant_floor',
        'in floor': 'radiant_floor',
        'mini split': 'mini_split',
        'individual rm controls': 'zoned',
        'perimeter': 'perimeter_radiant',  # 周边辐射，地暖变种
        'hot water': 'hydronic',  # 热水=水暖
        'gravity': 'gravity',  # 重力对流
        'wood / pellet': 'wood_pellet',
        'pellet stove': 'pellet_stove',
        'space heater': 'space_heater',
        'natural gas': 'gas',
        'propane / butane': 'propane',
        'gas plumbed': 'gas',
        'gas': 'gas',
        'electric': 'electric',
        'solar heat': 'solar',
        'geothermal': 'geothermal',
        'oil': 'oil',
        'kerosene': 'kerosene',
        'coal': 'coal',
        'steam': 'steam',
        'stove': 'stove',
        'fireplace insert': 'fireplace',
        'fireplace(s)': 'fireplace',
        'yes': 'unknown',  # 没有信息的yes
        'unknown': 'unknown',
        'see remarks': 'unknown',
        'none': 'none',
        'smart vent': 'smart_vent',
        'humidity control': 'humidity_control',
        'energy star qualified equipment': 'energy_star',
        'high efficiency': 'high_efficiency',
        'zoned': 'zoned',
        'combination': 'mixed_system',  # 组合系统
        'ductless': 'mini_split',  # 无管道=分体式
        'central': 'forced_air',  # 单独的central=强制通风
        'baseboard': 'baseboard',
        'radiant': 'radiant',
        'hydronic': 'hydronic',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # 3. 删除停用词
    text = re.sub(r'\b(and|or|the|a|an|in|on|with|by|is|to|for|of|s)\b', '', text)
    
    # 4. 去重空格
    text = ' '.join(text.split())
    
    return text.strip()

def clean_flooring_token(text):
    """Flooring专用清洗：统一材料名称，处理复合地板与品质描述"""
    if pd.isna(text):
        return ''
    
    text = str(text).lower()
    
    # 1. 删除数字前缀和编号
    text = re.sub(r'^\d+\s+', '', text)
    
    # 2. **统一核心材料词（带优先级）**
    replacements = {
        # 木材类（区分品质）
        'hardwood flrs throughout': 'hardwood_throughout',  # 全屋硬木，价值标记
        'hardwood': 'hardwood',
        'engineered wood': 'engineered_wood',
        'reclaimed wood': 'reclaimed_wood',
        'bamboo': 'bamboo',
        'softwood': 'softwood',
        'simulated wood': 'laminate',  # 仿木=复合地板
        'wood/wood like': 'wood_like',
        'wood/laminate': 'wood_laminate',
        
        # 瓷砖类（天然石材）
        'ceramic tile': 'tile',
        'mexican tile': 'tile',
        'stone tile': 'tile',
        'porcelain': 'tile',
        'slate': 'tile',
        'travertine': 'tile',
        'stone/travertine': 'tile',
        'natural stone': 'stone',
        
        # 复合地板
        'laminate wood': 'laminate',
        'laminated': 'laminate',
        'lvp luxury vinyl pl': 'vinyl',  # 高端Vinyl
        
        # Vinyl/Linoleum合并（同类材料）
        'linoleum / vinyl': 'vinyl',
        'vinyl / linoleum': 'vinyl',
        'linoleum/vinyl': 'vinyl',
        'vinyl/linoleum': 'vinyl',
        'sheet vinyl': 'vinyl',
        'vinyl tile': 'vinyl',
        'linoleum': 'vinyl',
        
        # 地毯类（处理部分铺设）
        'carpeted': 'carpet',
        'partial carpet': 'partial_carpet',  # 部分地毯=旧房标记
        'partial carpeting': 'partial_carpet',
        'partially carpeted': 'partial_carpet',
        'natural fiber carpet': 'carpet',
        'recycled carpet': 'carpet',
        
        # 硬质地面
        'stamped': 'stamped_concrete',
        'cement': 'concrete',
        'asphalt tile': 'asphalt',
        
        # 质量描述标准化
        'new flooring': 'new_flooring',
        'no flooring': 'unfinished',
        'needs to be replaced': 'unfinished',
        'varies by unit': 'varies',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # 3. 删除停用词和修饰词
    text = re.sub(r'\b(and|or|the|a|an|in|on|with|by|is|to|for|of|see|remarks|other|other-attch|other-rmks|s)\b', '', text)
    
    # 4. 删除多余空格
    text = ' '.join(text.split())
    
    return text.strip()

def extract_type_dimensions(text):
    """从混合字符串中提取7个维度"""
    if pd.isna(text):
        return []
    
    text = str(text).lower()
    
    # 1. 删除异常token（如"31", "SINGLE, AGRI/LV"等纯噪音）
    text = re.sub(r'^\d+,?\s*|single,\s*agri/lv', '', text)
    
    # 2. 按逗号分割
    tokens = [t.strip() for t in text.split(',') if t.strip()]
    
    # 3. 维度提取
    dimensions = []
    
    # 维度1: 建筑类型（必须提取，互斥）
    type_patterns = {
        'single_family': ['single family', 'singlefamily', 'ranch', 'detached'],
        'townhouse': ['townhouse'],
        'condo': ['condo', 'condominium', 'cooperative', 'apartment'],
        'duplex': ['duplex', 'triplex', 'multifamily', '2 houses', '2+ residences'],
        'manufactured': ['manufactured home', 'manufactured on land', 'mobile home', 'mobilemanufactured', 'mfd-f', 'double wide'],
        'planned_unit': ['planned unit development'],
        'vacant_land': ['vacantland', 'vacant land', 'residential lot'],
    }
    
    # 维度2: 结构（仅对single_family有效）
    structure_patterns = {
        'detached': ['detached'],
        'attached': ['attached'],
        'semi_attached': ['semi-attached', 'semi attached'],
    }
    
    # 维度3: 层数（Level=Story）
    floor_patterns = {
        '1_level': ['1 level', '1 story'],
        '2_level': ['2 level', '2 story', 'split level'],
        '3_level': ['3 level', '3 story'],
        '4_plus_level': ['4 story', '4+ story', '4+ level'],
    }
    
    # 维度4: 建筑高度
    height_patterns = {
        'low_rise': ['low-rise (1-3)', 'low-rise', 'ground floor', 'flat'],
        'mid_rise': ['mid-rise (4-8)', 'mid-rise', 'mid rise'],
        'hi_rise': ['hi-rise (9+)', 'hi-rise'],
    }
    
    # 维度5: 单元位置
    position_patterns = {
        'ground_floor': ['ground floor'],
        'top_floor': ['top floor'],
        'unit_above': ['unit above'],
        'unit_below': ['unit below'],
        'penthouse': ['penthouse'],
    }
    
    # 维度6: 特殊属性
    if 'luxury' in tokens:
        dimensions.append('luxury')
    if 'new construction' in text or 'new construction' in ' '.join(tokens):
        dimensions.append('new_construction')
    if 'live/work' in text:
        dimensions.append('live_work')
    
    # 7. 对每个token进行模式匹配
    for token in tokens:
        # 类型匹配（仅取第一个匹配）
        if not any(d.startswith('type_') for d in dimensions):
            for type_name, patterns in type_patterns.items():
                if any(pattern in token for pattern in patterns):
                    dimensions.append(f'type_{type_name}')
                    break
        
        # 结构匹配
        for struct_name, patterns in structure_patterns.items():
            if any(pattern in token for pattern in patterns):
                dimensions.append(f'structure_{struct_name}')
        
        # 层数匹配
        for floor_name, patterns in floor_patterns.items():
            if any(pattern in token for pattern in patterns):
                dimensions.append(f'floors_{floor_name}')
        
        # 高度匹配
        for height_name, patterns in height_patterns.items():
            if any(pattern in token for pattern in patterns):
                dimensions.append(f'height_{height_name}')
        
        # 位置匹配
        for pos_name, patterns in position_patterns.items():
            if any(pattern in token for pattern in patterns):
                dimensions.append(f'position_{pos_name}')
    
    # 8. 未知处理
    if not dimensions:
        return ['unknown']
    
    # 9. 去重并保持顺序
    seen = set()
    unique_dim = []
    for d in dimensions:
        if d not in seen:
            seen.add(d)
            unique_dim.append(d)
    
    return unique_dim

def map_features(clean_text, feature_map):
    """将清洗后的文本映射到标准类别"""
    if not clean_text:
        return []
    
    # 查找所有匹配的类别
    matched = []
    for category, keywords in feature_map.items():
        if any(keyword in clean_text for keyword in keywords): # 这里用的是生成器表达式，keyword in clean_text for keyword in keywords生成一串布尔序列，any函数检查是否有True
            matched.append(category)
    
    # 如果没匹配，返回通用标签
    return matched if matched else ['other']

def process_categorical_features(all_features):
    """处理类别特征，返回处理后的DataFrame"""
    # 1. 清洗原始文本
    Parking_clean = all_features['Parking features'].apply(clean_parking_token)
    Laundry_clean = all_features['Laundry features'].apply(clean_laundry_token)
    Appliances_clean = all_features['Appliances included'].apply(clean_appliances_token)
    Cooling_clean = all_features['Cooling features'].apply(clean_cooling_token)
    Heating_clean = all_features['Heating features'].apply(clean_heating_token)
    Flooring_clean = all_features['Flooring'].apply(clean_flooring_token)
    type_extract = all_features['Type'].apply(extract_type_dimensions)

    # 2. 语义合并，映射到标准类别
    Parking_categories = Parking_clean.apply(lambda x: map_features(x, PARKING_MAP))
    Laundry_categories = Laundry_clean.apply(lambda x: map_features(x, LAUNDRY_MAP))
    Appliances_categories = Appliances_clean.apply(lambda x: map_features(x, APPLIANCES_MAP))
    Cooling_categories = Cooling_clean.apply(lambda x: map_features(x, COOLING_MAP))
    Heating_categories = Heating_clean.apply(lambda x: map_features(x, HEATING_MAP))
    Flooring_categories = Flooring_clean.apply(lambda x: map_features(x, FLOORING_MAP))

    # 3. 多标签数据的独热编码（使用MultiLabelBinarizer，每个样本可以有多个标签）
    mlb = MultiLabelBinarizer()
    parking_encoded = pd.DataFrame(
        mlb.fit_transform(Parking_categories), # Parking_categories中的每一行都是包含多个类别标签的列表
        columns=[f'parking_{cat}' for cat in mlb.classes_], # classes_中包含Parking_categories中的所有类别
        index=all_features.index
    )
    laundry_encoded = pd.DataFrame(
        mlb.fit_transform(Laundry_categories),
        columns=[f'laundry_{cat}' for cat in mlb.classes_],
        index=all_features.index
    )
    appliances_encoded = pd.DataFrame(
        mlb.fit_transform(Appliances_categories),
        columns=[f'appliance_{cat}' for cat in mlb.classes_],
        index=all_features.index
    )
    cooling_encoded = pd.DataFrame(
        mlb.fit_transform(Cooling_categories),
        columns=[f'cooling_{cat}' for cat in mlb.classes_],
        index=all_features.index
    )
    heating_encoded = pd.DataFrame(
        mlb.fit_transform(Heating_categories),
        columns=[f'heating_{cat}' for cat in mlb.classes_],
        index=all_features.index
    )
    flooring_encoded = pd.DataFrame(
        mlb.fit_transform(Flooring_categories),
        columns=[f'flooring_{cat}' for cat in mlb.classes_],
        index=all_features.index
    )
    type_encoded = pd.DataFrame(
        mlb.fit_transform(type_extract),
        columns=[f'{cat}' for cat in mlb.classes_],  # 列名已带前缀，无需加type_
        index=all_features.index
    )

    # 4. 合并
    all_features = pd.concat([all_features, type_encoded, parking_encoded, laundry_encoded, 
                              appliances_encoded, cooling_encoded, heating_encoded, 
                              flooring_encoded], axis=1)
    # 5. 删除原列
    all_features.drop(columns=['Type', 'Parking features', 'Laundry features', 'Appliances included', 
                               'Cooling features', 'Heating features', 'Flooring'], inplace=True)
    
    return all_features

def process_Zip_features(all_features):
    """处理Zip码特征，将每一位数字独热编码，共5位×10=50个特征"""
    
    # 确保Zip为5位字符串，不足5位前面补0
    all_features['Zip'] = all_features['Zip'].astype(str)

    # 对每一位进行独热编码
    for pos in range(3):
        # 提取第pos位数字
        digit_col = all_features['Zip'].str[pos] # .str 提供类似 Python 字符串的方法，但作用于整个 Series，效率更高，返回一个series
        
        # 为该位创建10个特征（0-9）
        for d in range(10):
            col_name = f'zip_pos{pos}_digit{d}'
            all_features[col_name] = (digit_col == str(d)).astype(int)
    
    # 删除原始Zip列
    all_features.drop(columns=['Zip'], inplace=True)
    
    return all_features

def data_preprocess(train_data, test_data):
    BASE_DIR = Path(__file__).resolve().parent

    print(len(train_data), len(test_data))

    """处理房间数异常值"""
    train_data['Bedrooms'] = train_data['Bedrooms'].apply(roomstr_to_num)
    test_data['Bedrooms'] = test_data['Bedrooms'].apply(roomstr_to_num)

    """删除缺失值过多和不好填充的样本"""
    train_data = drop_samples(train_data)
    print(len(train_data), len(test_data))

    """删除无关列"""
    # 将训练数据和测试数据拼接在一起，丢弃train_data前3列，test_data前2列（这些列用不到）
    all_features = pd.concat((train_data.iloc[:, 3:], test_data.iloc[:, 2:]))
    # 删除其他不使用的列，axis=1表示提供的是列名
    all_features = all_features.drop(['Total spaces', 'Heating', 'Cooling', 'Parking', 'Region',
                                    'Elementary School', 'Middle School', 'High School',
                                    'City', 'State'], axis=1)
    
    """填充缺失值"""
    # Last Sold On和Last Sold Price为空意味着是新房，添加一个新的二元特征来标记是否为新房
    all_features['Is_New_Home'] = all_features['Last Sold Price'].isna().astype(int)

    # 按特征填充缺失值，由于上面添加了新房标记，这里可以对Last Sold On和Last Sold Price直接填充0
    all_features = all_features.fillna({
        'Summary': 'No Summary',
        'Year built': all_features['Year built'].median(),
        'Lot': all_features['Lot'].median(),
        'Bedrooms': all_features['Bedrooms'].median(),
        'Bathrooms': all_features['Bathrooms'].median(),
        'Full bathrooms': all_features['Full bathrooms'].median(),
        'Total interior livable area': all_features['Total interior livable area'].median(),
        'Garage spaces': 0,
        'Elementary School Score': all_features['Elementary School Score'].median(),
        'Elementary School Distance': all_features['Elementary School Distance'].median(),
        'Middle School Score': all_features['Middle School Score'].median(),
        'Middle School Distance': all_features['Middle School Distance'].median(),
        'High School Score': all_features['High School Score'].median(),
        'High School Distance': all_features['High School Distance'].median(),
        'Tax assessed value': all_features['Tax assessed value'].median(),
        'Annual tax amount': all_features['Annual tax amount'].median(),
        'Flooring': 'Unknown',
        'Heating features': 'Unknown',
        'Cooling features': 'Unknown',
        'Appliances included': 'Unknown',
        'Laundry features': 'Unknown',
        'Parking features': 'Unknown',
        'Last Sold Price': 0 # 没有说明是新房，填0
    })

    """特征工程"""
    """处理时间特征"""
    # 将时间特征转换为数值特征（距离2099-01-01的天数/年数）
    reference_date = pd.to_datetime('2099-01-01')

    all_features['Last Sold On'] = pd.to_datetime(all_features['Last Sold On'], errors='coerce')
    all_features['Listed On'] = pd.to_datetime(all_features['Listed On'], errors='coerce')

    all_features['Days_Since_Last_Sold'] = (reference_date - all_features['Last Sold On']).dt.days
    all_features['Days_Since_Listed'] = (reference_date - all_features['Listed On']).dt.days
    all_features['House_Age'] = reference_date.year-all_features['Year built']

    all_features['Days_Since_Last_Sold'] = all_features['Days_Since_Last_Sold'].fillna(0) # 填充缺失值为0
    all_features['Days_Since_Listed'] = all_features['Days_Since_Listed'].fillna(0) # 填充缺失值为0
    
    all_features.drop(columns=['Year built', 'Last Sold On', 'Listed On'], inplace=True)

    """调用大模型为Summary打分"""
    # scores_df = llm_caller.get_house_scores(all_features['Summary'])

    # # 检查索引唯一性，这里的一个坑是如果DataFrame里有多个样本的索引相同，并且通过idx进行设值时，就会同时设置到多个样本上，对于不存在的idx还会创建新行保存。
    # # 因为scores_df的索引是唯一的，而all_features的索引可能不是唯一的。
    # if not all_features.index.is_unique: 
    #     all_features = all_features.reset_index(drop=True)

    # # 这里不用idx设值以避坑，直接使用 concat 进行横向合并
    # scores_df.columns = [f'summary_{col}' for col in scores_df.columns]
    # all_features = pd.concat([all_features, scores_df], axis=1)

    all_features.drop(columns=['Summary'], inplace=True)

    """数值特征标准化"""
    numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
    numeric_features = numeric_features.drop(['Is_New_Home','Zip','Tax assessed value',
                           'Listed Price','Last Sold Price'])  # 标签特征不标准化

    features_mean_std = pd.DataFrame(all_features[numeric_features].mean(), columns=['mean'])
    features_mean_std['std'] = all_features[numeric_features].std()
    features_mean_std.to_csv(BASE_DIR.joinpath('csv/features_mean_std.csv')) # 保存均值和标准差以备后续推理使用

    all_features[numeric_features] = all_features[numeric_features].apply( # apply作用于每一列
        lambda x: (x - x.mean()) / (x.std()) # x为列向量
    )

    """价格相关特征对数变换"""
    price_related_features = ['Tax assessed value',
                              'Listed Price', 'Last Sold Price']
    all_features[price_related_features] = all_features[price_related_features].apply(
        lambda x: np.log1p(x) 
    )

    """处理类别特征：清洗、语义合并、独热编码"""
    all_features = process_categorical_features(all_features)

    """处理Zip码类别特征：分位数字独热编码"""
    all_features = process_Zip_features(all_features)

    all_features.to_csv(BASE_DIR.joinpath('csv/all_features_processed_LLM.csv'), index=False) # 保存处理后的特征

    return all_features, train_data