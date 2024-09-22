import logging
from typing import Optional
from fastapi import Depends
from pydantic import BaseModel
from modules.plugins import PluginBase, release_plugin, use_plugin
from plugins.exllamav2 import ExllamaV2Plugin
from utils.text_utils import json_from_chat


class WebcrawlRequest(BaseModel):    
    prompt: str
    data: str    


@PluginBase.router.post("/txt/answer", tags=["Text Generation"])
async def crawl(req: WebcrawlRequest):

    messages = [
        {
            "role": "system",
            "content": f"You are a pretending to be a program that only outputs JSON (don't blow your cover!). You will be provided with some data followed by a request from the user. Your job is to look at the data and create a JSON formatted object with the exact properties requested by the user. At the end of the JSON include [END] so we know where the actual JSON ends. Here is the data:\n\n{req.data}\nPlease await a request from the user, and remember to only use the supplied data when answering. Never make up answers not based on the supplieid data. Here is the data:\n\n{req.data}",
        },
        {
            "role": "user",
            "content": req.prompt,
        },
    ]

    plugin = None

    try:
        plugin: ExllamaV2Plugin = await use_plugin(ExllamaV2Plugin)
        response = "".join([x async for x in plugin.generate_chat_response(
            messages=messages, max_new_tokens=1000
        )])
        print(response)
        obj = json_from_chat(response)
        return obj

    except Exception as e:
        logging.error(e, exc_info=True)
        raise e

    finally:
        if plugin is not None:
            release_plugin(plugin)


@PluginBase.router.get("/txt/webcrawl", tags=["Text Generation"])
async def crawl_get(req: WebcrawlRequest = Depends()):
    return await crawl(req)


class TxtLawRequest(BaseModel):
    prompt: str
    title: Optional[int] = None
    chapter: Optional[int] = None


titles = """
Title I CONSTRUCTION OF STATUTES (Ch. 1-2)
Title II STATE ORGANIZATION (Ch. 6-8)
Title III LEGISLATIVE BRANCH; COMMISSIONS (Ch. 10-11)
Title IV EXECUTIVE BRANCH (Ch. 14-24)
Title V JUDICIAL BRANCH (Ch. 25-44)
Title VI CIVIL PRACTICE AND PROCEDURE (Ch. 45-88)
Title VII EVIDENCE (Ch. 90-92)
Title VIII LIMITATIONS (Ch. 95-95)
Title IX ELECTORS AND ELECTIONS (Ch. 97-107)
Title X PUBLIC OFFICERS, EMPLOYEES, AND RECORDS (Ch. 110-122)
Title XI COUNTY ORGANIZATION AND INTERGOVERNMENTAL RELATIONS (Ch. 124-164)
Title XII MUNICIPALITIES (Ch. 165-185)
Title XIII PLANNING AND DEVELOPMENT (Ch. 186-191)
Title XIV TAXATION AND FINANCE (Ch. 192-220)
Title XV HOMESTEAD AND EXEMPTIONS (Ch. 222-222)
Title XVI TEACHERS' RETIREMENT SYSTEM; HIGHER EDUCATIONAL FACILITIES BONDS (Ch. 238-243)
Title XVII MILITARY AFFAIRS AND RELATED MATTERS (Ch. 250-252)
Title XVIII PUBLIC LANDS AND PROPERTY (Ch. 253-274)
Title XIX PUBLIC BUSINESS (Ch. 279-290)
Title XX VETERANS (Ch. 292-296)
Title XXI DRAINAGE (Ch. 298-298)
Title XXII PORTS AND HARBORS (Ch. 308-315)
Title XXIII MOTOR VEHICLES (Ch. 316-324)
Title XXIV VESSELS (Ch. 326-328)
Title XXV AVIATION (Ch. 329-333)
Title XXVI PUBLIC TRANSPORTATION (Ch. 334-349)
Title XXVII RAILROADS AND OTHER REGULATED UTILITIES (Ch. 350-368)
Title XXVIII NATURAL RESOURCES; CONSERVATION, RECLAMATION, AND USE (Ch. 369-380)
Title XXIX PUBLIC HEALTH (Ch. 381-408)
Title XXX SOCIAL WELFARE (Ch. 409-430)
Title XXXI LABOR (Ch. 435-452)
Title XXXII REGULATION OF PROFESSIONS AND OCCUPATIONS (Ch. 454-493)
Title XXXIII REGULATION OF TRADE, COMMERCE, INVESTMENTS, AND SOLICITATIONS (Ch. 494-560)
Title XXXIV ALCOHOLIC BEVERAGES AND TOBACCO (Ch. 561-569)
Title XXXV AGRICULTURE, HORTICULTURE, AND ANIMAL INDUSTRY (Ch. 570-604)
Title XXXVI BUSINESS ORGANIZATIONS (Ch. 605-623)
Title XXXVII INSURANCE (Ch. 624-651)
Title XXXVIII BANKS AND BANKING (Ch. 655-667)
Title XXXIX COMMERCIAL RELATIONS (Ch. 668-688)
Title XL REAL AND PERSONAL PROPERTY (Ch. 689-723)
Title XLI STATUTE OF FRAUDS, FRAUDULENT TRANSFERS, AND GENERAL ASSIGNMENTS (Ch. 725-727)
Title XLII ESTATES AND TRUSTS (Ch. 731-740)
Title XLIII DOMESTIC RELATIONS (Ch. 741-753)
Title XLIV CIVIL RIGHTS (Ch. 760-765)
Title XLV TORTS (Ch. 766-774)
Title XLVI CRIMES (Ch. 775-896)
Title XLVII CRIMINAL PROCEDURE AND CORRECTIONS (Ch. 900-985)
Title XLVIII EARLY LEARNING-20 EDUCATION CODE (Ch. 1000-1013)
Title XLIX PARENTS' BILL OF RIGHTS; TEACHERS' BILL OF RIGHTS (Ch. 1014-1015)
"""

@PluginBase.router.post("/law/fl", tags=["Text Generation"])
async def florida_law(req: TxtLawRequest):    

    req = WebcrawlRequest(        
        prompt="Here are a list of Title sections for the law in the state of Florida. You must determine the most relevant statute for the given prompt. I need a property called title with the full line of text exactly as shown for the title most related to the following, and please convert the roman numerals to integers:\n\n" + req.prompt + "\n\nYou may give your response now. Output as a single JSON object with string property values. Begin your output now.",
        data=titles
    )
    return await crawl(req)


@PluginBase.router.get("/law/fl", tags=["Text Generation"])
async def florida_law_get(req: TxtLawRequest = Depends()):
    return await florida_law(req)
