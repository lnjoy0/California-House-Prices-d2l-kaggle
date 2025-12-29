import asyncio
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from tqdm.asyncio import tqdm_asyncio
import pandas as pd
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from typing import List, Dict, Any
import llm_api_key
from datetime import datetime

# 定义Pydantic
class Scores(BaseModel):
    attractiveness_score: float = Field(description="Score 0-10 for emotional appeal")
    completeness_score: float = Field(description="Score 0-10 for specs detail")
    condition_score: float = Field(description="Score 0-10 for physical condition")
    readability_score: float = Field(description="Score 0-10 for grammar/tone")
    # reasoning: str = Field(description="Brief explanation, max 30 words") # 输出token越少越便宜

SYSTEM_TEMPLATE = """
# Role
You are an expert Real Estate Appraiser and Data Scientist with 20 years of experience. Your objective is to analyze real estate listing descriptions (summaries) and extract quantifiable feature scores to assist a machine learning model in predicting house prices.

# Task
Read the provided "Summary" text and score it on four distinct dimensions using a scale of 0 to 10. The output must be strictly structured for programmatic ingestion.

# Scoring Criteria

**1. Attractiveness Score (`attractiveness_score`)**
* **Definition:** Evaluates the emotional appeal, sales pressure, and marketing sentiment.
* **0-3 (Low):** Dry, negative, or neutral language. Keywords: "as-is", "motivated seller", or boring factual lists without adjectives.
* **4-6 (Medium):** Standard description. Objective and functional but lacks enthusiasm.
* **7-8 (High):** Positive adjectives used (e.g., "beautiful", "spacious", "charming").
* **9-10 (Very High):** Highly persuasive, emphasizing scarcity or lifestyle. Keywords: "Must see!", "Rare opportunity", "Breathtaking views".

**2. Completeness Score (`completeness_score`)**
* **Definition:** Evaluates the density of factual information (specs) provided.
* **0-3 (Low):** Vague; missing basic details like bed/bath counts or parking.
* **4-6 (Medium):** Mentions basic specs (bed/bath) but lacks detail on amenities or layout.
* **7-8 (High):** Details rooms, garage/parking, lot features, and potential community amenities.
* **9-10 (Very High):** Comprehensive audit of the property, including specific layout details, school districts, nearby transport, and dimensions.

**3. Condition & Upgrades Score (`condition_score`)**
* **Definition:** Evaluates the physical state of the house and recent renovations. **This is the most critical factor for price.**
* **0-3 (Low):** Needs work. Keywords: "TLC needed", "Fixer-upper", "Handyman special", "Dated".
* **4-6 (Medium):** Average condition or undefined. No specific mention of recent updates.
* **7-8 (High):** Partial updates mentioned. Keywords: "New paint", "New carpet", "Updated fixtures".
* **9-10 (Very High):** Turn-key / Luxury condition. Keywords: "Fully remodeled", "Brand new kitchen", "Stainless steel appliances", "New roof/AC/Windows".

**4. Readability & Professionalism (`readability_score`)**
* **Definition:** Evaluates the grammar, formatting, and professional tone.
* **0-3 (Low):** "Spammy" appearance. Excessive use of ALL CAPS, multiple exclamation marks (!!!), ellipses (....), or typos. Hard to read.
* **4-6 (Medium):** Fragmented sentences or bullet-point style lists.
* **7-8 (High):** Complete sentences, good grammar, easy flow.
* **9-10 (Very High):** Elegant, professional copywriting style suitable for a luxury brochure.

# Output Format
{format_instructions}

# Important
Return ONLY the JSON object. No markdown code blocks.
"""

async def process_single_summary(chain, summary_text: str) -> Dict[str, Any]: # async声明异步携程函数，调用时需要带await，或者asyncio.run等
    """处理单条数据，包含错误处理"""
    try:
        # ainvoke 自动执行 Prompt -> LLM -> Parser
        result = await chain.ainvoke({"summary_text": summary_text})
        # 将 Pydantic 对象转为字典
        return result.dict()
    except Exception as e:
        # 发生错误时返回默认值，避免程序中断
        print(f"Error processing: {summary_text[:30]}... | {e}")
        return {
            "attractiveness_score": 0.0,
            "completeness_score": 0.0,
            "condition_score": 0.0,
            "readability_score": 0.0,
            #"reasoning": f"Error: {str(e)[:50]}"
        }

async def batch_process(llm, summaries: List[str], concurrency: int): # 函数内部有 await，因此在语法上必须为 async def，即使batch_call本身不需要异步调用
    """批量并发处理"""

    # 初始化解析器
    parser = PydanticOutputParser(pydantic_object=Scores)

    # 1. 创建 Chain (Prompt | LLM | Parser)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_TEMPLATE),
        ("user", "{summary_text}")
    ]).partial(format_instructions=parser.get_format_instructions())
    
    chain = prompt | llm | parser

    # 2. 限制并发数的信号量
    semaphore = asyncio.Semaphore(concurrency)

    async def bounded_call(text):
        async with semaphore:
            return await process_single_summary(chain, text)

    # 3. 创建任务列表
    tasks = [bounded_call(text) for text in summaries]

    # 4. 执行并显示进度条
    print(f"开始处理 {len(tasks)} 条数据，并发数: {concurrency}...")
    results = await tqdm_asyncio.gather(*tasks)
    return results

def get_house_scores(df_series):
    """
    同步包装器：负责初始化 LLM 并运行异步循环
    """

    # 初始化模型
    llm = ChatOpenAI(
        model = "deepseek-chat",
        openai_api_key = llm_api_key.ds_key,
        openai_api_base = "https://api.deepseek.com",
        temperature = 0.0,
        max_retries = 3
    )

    # 运行异步事件循环
    # 注意：如果在 Jupyter Notebook 中运行，asyncio.run() 可能会报错，
    # 在 Jupyter 中请直接使用 await batch_process(...)
    # 如果不在jupyter中运行可以直接使用results = asyncio.run(batch_process(llm, df_series.values))，不用获取或创建事件循环
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    if loop.is_running():
        # Jupyter 环境处理
        import nest_asyncio
        nest_asyncio.apply()
        results = loop.run_until_complete(batch_process(llm, df_series.values, concurrency=20))
    else:
        # 普通脚本环境
        results = asyncio.run(batch_process(llm, df_series.values, concurrency=20))

    # 合并结果回 DataFrame
    scores_df = pd.DataFrame(results)

    sv_scores_df = pd.concat([df_series.reset_index(drop=True), scores_df], axis=1)
    sv_scores_df.to_csv("llm_scores_"+str(datetime.now()).replace(':','.')+".csv", index=False) # 保存打分结果以备检查

    return scores_df

# if __name__ == "__main__":
#     test_example = ["15994 3rd St, Snelling, CA 95369 is a single family home that contains 1,272 sq ft and was built in 1947. It contains 2 bedrooms and 2 bathrooms. This home last sold for $185,000 in January 2020. The Zestimate for this house is $230,302. The Rent Zestimate for this home is $1,375/mo.",
#                     # "Very fancy house and quiet neighborhood.Living room, dining room, master room, patio with basic furniture. TV, refrigerator and wash/dryer included. Owner pays for garbage and sewer, tenant pays for PG&E, water, internet. 6 month lease or longer.",
#                     # "Rarely available prime location!  With 1640 square feet of renovated living space this home featuring a flexible floor plan which offers a comfortable and relaxed lifestyle. Whether you prefer to stroll the one block walk to the best Pleasure Point shops, cafes & restaurants or you are heading straight to the world class beaches and surf spots this location will not disappoint.  This lovely 3 Bedroom, 2 Bathroom home features an inviting front patio space to host old friends and make new ones.  With it's over sized Living Room remodeled Kitchen and the large Master Suite retreat features a spacious walk in closet, attached office space and access to a private side yard deck. Featuring an attached two car garage PLUS bonus storage area for your surf shack, bikes, kayaks, arts and crafts or practically any hobby you will not feel cramped here.  If you are like me, you are sure to enjoy this friendly and inviting neighborhood.  Don't miss out, make this lovely Opal Cliffs home yours!",
#                     # "Outstanding opportunity to own a single level updated duplex in one of the strongest rental markets in the Nation. Desirable unit mix of (2) 2bd/1bth units. Each unit has a private rear yard, single car garage, fireplace, forced air heating, dual pane windows, & individual water heaters.  Convenient central location near Downtown Campbell with quick access to highways. Both units have a functional floor-plan with lots of natural light.  One of the units is vacant & ready for a tenant at market rent.",
#                     # "Welcome to this lovely home in the well sought after Los Paseos neighborhood. This beautiful home has open floor plan with bright natural lighting.  Updated spacious kitchen will be a great place for wonderful gatherings and conversations. Home features 3 beds, 2 full baths,  Dual Pane Windows, Water Softener, Recess Lighting, Crown Molding and updated Laminate Flooring.  Home is conveniently close to School, Park, Shopping, 101, 85 freeways.  Relax in a well maintained backyard and enjoy some seasonal fruits.  This is a wonderful home not to be missed.",
#                     # "Beautifully upgraded one bedroom condo situated within minutes of downtown Burlingame, CalTrain station and Highway 101.  Enjoy your privacy in this single story, spacious end unit with only one attached neighbor, located close to the elevator.  Featuring an open, bright, newly painted living area with warm designer colors, wood-style laminate & carpet flooring, crown moldings and plantation shutters.  Dining area/kitchen updates includes stainless steel appliances, granite countertops and backsplash.  Bathroom has new vanity/sink and fixtures.  This small 16-unit complex includes a sparkling swimming pool, spa, recreation room, laundry on-site, secured building access, dedicated secure parking space, and extra storage closet.  This condo is perfect to start out or downsizing or investing.Its central location is uniquely convenient: you can walk to Caltrain, and downtown Burlingame shopping.",
#                     # "REDUCED!  Priced to sell.  Beautiful remodeled duplex in prime Beverly Hills. The property is immediately adjacent to Beverly Hills High School, Roxbury Park, The Peninsula Hotel & Rodeo Drive. Upper Unit, w/3Br+2.5Ba, large living rm w/fireplace, formal dining room, eat-in kitchen, & laundry rm w/washer & dryer. Generous master suite & 2 additional good-sized bedrooms. Unit will be delivered vacant. Downstairs Unit, has a grand entry, large living room w/fireplace, formal dining room & 2 master suites. Gourmet eat-in kitchen w/Sub Zero, & granite counters. Hardwood floors, central A/C. Each Unit has a 2-car garage & private outdoor area. Beverly Hills Unified School District.",
#                     # "A must see home located within a desirable neighborhood in North San Jose. This home features 4 bedrooms, 2.5 bathrooms, spacious lot, two car garage, and various unimaginable details throughout. This property displays beautiful solid hardwood flooring, venetian plaster along staircase to match beautifully with the crystal chandeliers above the living room and dining room. Kitchen has beautiful tile backsplash with high end KitchenAid Refrigerator and Viking appliances. Each bedroom has intricate high end finishes that are to die for. Backyard offers a peaceful and private area perfect for entertaining guests.",
#                     # "Willow Glen Charmer!  This wonderful floor plan welcomes you through custom front doors into 5br and 2.5 ba in 2479sf. Entertain with ease in a remodeled chefs kitchen boasting cherry cabinets, granite counters, large center island with copper prep sink, modern pendant lighting and large convection range. Kitchen cabinets have pull out shelves and soft close doors. The kitchen flows into the eating area, with ample built in desk space. Adjacent family room has sliding glass doors leading to a beautiful covered deck, perfect for dining al fresco.  Downstairs has both a formal living room with fireplace and dining room. Remodeled half bath. Spacious indoor laundry room.  Hardwood floors. Master bath has shower.  Main hall bath has tub.  Cellular blinds on dual paned windows.  Abundant storage areas.  Fruit trees in backyard. Membership to neighborhood Brighton Square Pool.  Conveniently located minutes from major commuting routes & near many shopping, dining, & entertainment options.",
#                     "Bright and airy two bedroom, two bathroom condo with an open design.  When you enter the living room, you'll notice the high ceilings and hardwood floors. The kitchen has upgraded granite counter tops and plenty of cabinet space. The unit has a private balcony, intercom system, and a washer and dryer located inside. Community offers an association swimming pool with low HOA fee. Conveniently located in the center of Korea town."]
#     df_test = pd.DataFrame(test_example, columns=['Summary'])
#     scores_df = get_house_scores(df_test['Summary'])
#     print(scores_df)