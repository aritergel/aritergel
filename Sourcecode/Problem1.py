import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

BASE_URL = "https://fbref.com"
SEASON_URL = f"{BASE_URL}/en/comps/9/Premier-League-Stats"


TABLE_IDS = [
    "stats_standard", "stats_goalkeeping", "stats_shooting", "stats_passing",
    "stats_passing_types", "stats_gca", "stats_defense", "stats_possession",
    "stats_misc"
]

def get_player_urls():
    res = requests.get(SEASON_URL)
    soup = BeautifulSoup(res.text, 'lxml')
    stats_table = soup.find("table", {"id": "stats_standard"})
    links = stats_table.find_all("a")
    player_urls = set()
    for link in links:
        href = link.get("href", "")
        if "/en/players/" in href:
            full_url = BASE_URL + href
            player_urls.add(full_url)
    return list(player_urls)

def parse_table(soup, table_id):
    try:
        table = soup.find("table", {"id": table_id})
        df = pd.read_html(str(table))[0]
        df.columns = df.columns.droplevel()
        return df
    except:
        return pd.DataFrame()

def collect_player_data(url):
    res = requests.get(url)
    soup = BeautifulSoup(res.text, 'lxml')
    player_data = {}
    
    for table_id in TABLE_IDS:
        df = parse_table(soup, table_id)
        if df.empty: continue
        row = df.iloc[-1]  # most recent season
        for col in row.index:
            value = row[col]
            if pd.isna(value): value = "N/a"
            player_data[f"{table_id}_{col}"] = value
    # Нэр, үндэстэн, баг, байрлал, нас
    name = soup.find('h1').text.strip()
    player_data["Name"] = name

    try:
        info_box = soup.select_one("div[data-template='Partials/Player/summary']")
        nation = info_box.select_one("img.flag").get("title", "N/a")
        position = info_box.text.split("Position:")[-1].split("\n")[0].strip()
        age = info_box.text.split("Age:")[-1].split("\n")[0].strip().split("(")[0].strip()
        player_data["Nation"] = nation
        player_data["Position"] = position
        player_data["Age"] = age
    except:
        player_data["Nation"] = player_data["Position"] = player_data["Age"] = "N/a"

    return player_data

def main():
    all_data = []
    print("Fetching player URLs...")
    player_urls = get_player_urls()
    print(f"Found {len(player_urls)} players. Collecting data...")

    for i, url in enumerate(player_urls):
        try:
            data = collect_player_data(url)
            all_data.append(data)
            print(f"[{i+1}/{len(player_urls)}] Collected: {data['Name']}")
        except Exception as e:
            print(f"Failed to collect from {url} due to {e}")
        time.sleep(1.5)  # avoid rate-limiting

    df = pd.DataFrame(all_data)
    df = df[df['stats_standard_Min'].apply(lambda x: float(x) if x != "N/a" else 0) > 90]
    df = df.sort_values(by='Name')
    df.to_csv("results.csv", index=False)
    print("Saved to results.csv")

if __name__ == "__main__":
    main()

