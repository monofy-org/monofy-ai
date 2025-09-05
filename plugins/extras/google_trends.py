import feedparser
from modules.plugins import router


async def get_trends():
    url = "https://trends.google.com/trending/rss?geo=US"
    feed = feedparser.parse(url)
    print(feed)
    trends = []
    for entry in feed.entries:
        item = {
            "title": entry.title,
            "link": entry.ht_news_item_url,
        }
        print(item)
        trends.append(item)

    return trends


@router.get("/google/trends")
async def google_trends():
    return await get_trends()
