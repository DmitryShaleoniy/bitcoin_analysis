import yfinance as yf
import pandas as pd
from datetime import datetime as dt, timedelta
import datetime as datet
import os
import requests


print("\nНАЧАЛО ПАРСИНГА ДАННЫХ")
print("\nПАРСИНГ OCVD") #yahoo finance
#парсер 

having = pd.read_csv("./data/csv/btc_no_vibrosi.csv")
having.reset_index()

last_csv = having["date"].iloc[-1]

last_dt = dt.strptime(last_csv, "%Y-%m-%d")
#end_dt = dt.utcnow().date() + timedelta(days=1)
#end_dt = end_dt.date()

#print(last_dt.date())
#print(dt.now().date())


if (dt.now().date() != last_dt.date()):
    #last_dt += timedelta(days=1)
    print(f"запрос данных с: {last_dt.date()}")

    btc = yf.download("BTC-USD", start=last_dt.date() ,interval="1d", multi_level_index=False)
    if not btc.empty:
        btc = btc.reset_index()

        btc['Date'] = pd.to_datetime(btc['Date']).dt.strftime('%Y-%m-%d')

        df = btc[['Open', 'Close', 'Volume', 'Date']].copy()
        df.columns = ["open", "close", "volume", "date"]

        new_rows_btc = df[df['date'] > last_csv].copy() #вторая проверка

        no_vib_path = "./data/csv/btc_no_vibrosi.csv"
        if not new_rows_btc.empty:

            if (os.path.exists(no_vib_path)):
                try:
                    new_rows_btc.to_csv(no_vib_path, mode='a', header=False, index=False)
                    print("\nOCVD no_vibrosi_csv успешно обновлен\nпоследняя записанная строка:")
                    print(pd.read_csv(no_vib_path).tail(1))
                except Exception as e:
                    print(f"ошибка btc_no_vibrosi: {e}")
            else:
                print(f"{no_vib_path} не существует")
        else:
            print("новых дней для записи не найдено")
    else:
        print("не удалось получить данные от yahoo")
    last_dt -= timedelta(days=1)
else:
    print("OCVD укомплектован")


#парсер юани

print("\nПАРСИНГ CNY-USD") #yahoo finance

cny_path = "./data/csv/china_apply.csv"

if(os.path.exists(cny_path)):
    last_csv = pd.read_csv(cny_path)["Date"].iloc[-1]
    last_dt = dt.strptime(last_csv, "%Y-%m-%d")
    if (dt.now().date() != last_dt.date()):
        last_dt += timedelta(days=1)
        print(f"запрос данных с {last_dt}")
        cny = yf.download("CNY=X", start=last_dt.date() ,interval="1d", multi_level_index=False)

        if not cny.empty:
            cny = cny.reset_index()
            cny['Date'] = pd.to_datetime(cny['Date']).dt.strftime('%Y-%m-%d')
            df = cny[['Date', 'Close']].copy()
            df.columns = ["Date", "Value"]
            new_rows = df[df['Date'] > last_csv].copy()
            if not new_rows.empty:
                try:
                    new_rows.to_csv(cny_path, mode='a', header=False, index=False)
                    print("\nCNY-USD csv успешно обновлен\nпоследняя записанная строка:")
                    print(pd.read_csv(cny_path).tail(1))
                except Exception as e:
                    print(f"ошибка cny: {e}")
        else:
            print("ничего не было загруженно")


        last_dt -= timedelta(days=1)
    else:
        print("YOAN-USD укомплектован")
else:
    print(f"{cny_path} не существует")

#print("\nПАРСИНГ RUB-USD")

print("ПАРСИНГ ИНДЕКСА GESI") #yahoo finance

gesi_path = "./data/csv/gesi.csv"

if(os.path.exists(gesi_path)):
    last_csv = pd.read_csv(gesi_path)["date"].iloc[-1]
    last_dt = dt.strptime(last_csv, "%Y-%m-%d")
    if (dt.now().date() != last_dt.date()):
        last_dt += timedelta(days=1)
        print(f"запрос данных с {last_dt}")
        gesi = yf.download("GESI", start=last_dt.date() ,interval="1d", multi_level_index=False)
        if not gesi.empty:
            gesi = gesi.reset_index()
            gesi['Date'] = pd.to_datetime(gesi['Date']).dt.strftime('%Y-%m-%d')
            df = gesi[['Date', 'Close']].copy()
            df.columns = ["date", "gesi_value"]
            new_rows = df[df['date'] > last_csv].copy()
            #сейчас нужно продублировать последнее значение до текущей даты, потому что индекс GESI не обновляется каждый день
            last_value = pd.read_csv(gesi_path)["gesi_value"].iloc[-1]
            today = dt.now().date()
            date_range = pd.date_range(start=last_dt.date(), end=today)
            for date in date_range:
                date_str = date.strftime('%Y-%m-%d')
                if date_str not in new_rows['date'].values:
                    new_rows = pd.concat([new_rows, pd.DataFrame({'date': [date_str], 'gesi_value': [last_value]})], ignore_index=True)
            new_rows = new_rows.sort_values(by='date').reset_index(drop=True)
            new_rows = new_rows[new_rows['date'] > last_csv].copy()
            if not new_rows.empty:
                try:
                    new_rows.to_csv(gesi_path, mode='a', header=False, index=False)
                    print("\nGESI csv успешно обновлен\nпоследняя записанная строка:")
                    
                    print(pd.read_csv(gesi_path).tail(1))
                except Exception as e:
                    print(f"ошибка gesi: {e}")
        else:
            print("ничего не было загружено")
        last_dt -= timedelta(days=1)
else:
    print(f"{gesi_path} не существует")

print("\nПАРСИНГ КОЛИЧЕСТВА АКТИВНЫХ АДРЕСОВ")

FILENAME = "./data/csv/btc_active_addresses.csv"
URL = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics" #нашел комьюнити апи с сайта коин метрикс
HEADERS = {"User-Agent": "Mozilla/5.0"}

def update_btc_addresses():
    start_time = "2009-01-03"

    if os.path.exists(FILENAME) and os.path.getsize(FILENAME) > 0: #раньше адреса были в json, перевел в более удобный CSV
        existing_df = pd.read_csv(FILENAME)
        if not existing_df.empty and "date" in existing_df.columns:
            start_time = str(existing_df["date"].max())
    else:
        existing_df = pd.DataFrame(columns=["date", "value"])

    all_records = []
    #page_token = None

    while True:
        params = {
            "assets": "btc",
            "metrics": "AdrActCnt",
            "frequency": "1d",
            "page_size": "10000",
            "start_time": start_time,
        }
        # if page_token:
        #     params["page_token"] = page_token

        response = requests.get(URL, params=params, headers=HEADERS)
        if response.status_code != 200: #если 200 то ОК
            print(f"ошибка получения данных активных адрессов coinmetrics: {response.status_code}")
            return

        res_json = response.json()
        records = res_json.get("data", [])
        all_records.extend(records)

        page_token = res_json.get("next_page_token")
        if not page_token:
            break

    if not all_records:
        print("новых данных нет.")
        return

    new_df = pd.DataFrame(all_records).dropna(subset=["AdrActCnt"])
    new_df["date"] = pd.to_datetime(new_df["time"]).dt.strftime("%Y-%m-%d")
    new_df["value"] = pd.to_numeric(new_df["AdrActCnt"]).astype(int)
    new_df = new_df[["date", "value"]]

    final_df = pd.concat([existing_df, new_df]).drop_duplicates(
        subset=["date"], keep="last"
    )
    final_df = final_df.sort_values(by="date").reset_index(drop=True)

    final_df.to_csv(FILENAME, index=False)
    print(
        f"данные по активным обновлены в {FILENAME}. все прошло хорошо"
    )

update_btc_addresses()

FILENAME = "./data/csv/btc_hash_rate.csv"

print("\nПАРСИНГ ХЕШ РЕЙТ")

def update_btc_hash_rate():
    start_time = "2009-01-03"

    if os.path.exists(FILENAME) and os.path.getsize(FILENAME) > 0: #раньше адреса были в json, перевел в более удобный CSV
        existing_df = pd.read_csv(FILENAME)
        if not existing_df.empty and "date" in existing_df.columns:
            start_time = str(existing_df["date"].max())
    else:
        existing_df = pd.DataFrame(columns=["date", "value"])

    all_records = []
    #page_token = None

    while True:
        params = {
            "assets": "btc",
            "metrics": "HashRate",
            "frequency": "1d",
            "page_size": "10000",
            "start_time": start_time,
        }
        # if page_token:
        #     params["page_token"] = page_token

        response = requests.get(URL, params=params, headers=HEADERS)
        if response.status_code != 200: #если 200 то ОК
            print(f"ошибка получения данных hash rate coinmetrics: {response.status_code}")
            return

        res_json = response.json()
        records = res_json.get("data", [])
        all_records.extend(records)

        page_token = res_json.get("next_page_token")
        if not page_token:
            break

    if not all_records:
        print("новых данных нет.")
        return

    new_df = pd.DataFrame(all_records).dropna(subset=["HashRate"])
    new_df["date"] = pd.to_datetime(new_df["time"]).dt.strftime("%Y-%m-%d")
    new_df["value"] = pd.to_numeric(new_df["HashRate"]).astype(int)
    new_df = new_df[["date", "value"]]

    final_df = pd.concat([existing_df, new_df]).drop_duplicates(
        subset=["date"], keep="last"
    )
    final_df = final_df.sort_values(by="date").reset_index(drop=True)

    final_df.to_csv(FILENAME, index=False)
    print(
        f"данные по hash rate обновлены в {FILENAME}. все прошло хорошо"
    )

update_btc_hash_rate()

#примечаение! тут хеш рейт немного в других единицах измерения, но мы производим полуную замену поэтому поменяться 
#ничего не должно, у нас просто теперь нет одного коэффициента перед значением хеш рейта. 


def coin_metrics_get_csv(filename, metric, start_time = "2009-01-03"):
    #start_time = "2009-01-03"

    FILENAME = filename

    if os.path.exists(FILENAME) and os.path.getsize(FILENAME) > 0: #раньше адреса были в json, перевел в более удобный CSV
        existing_df = pd.read_csv(FILENAME)
        if not existing_df.empty and "date" in existing_df.columns:
            start_time = str(existing_df["date"].max())
    else:
        existing_df = pd.DataFrame(columns=["date", "value"])

    all_records = []
    #page_token = None

    while True:
        params = {
            "assets": "btc",
            "metrics": metric,
            "frequency": "1d",
            "page_size": "10000",
            "start_time": start_time,
        }
        # if page_token:
        #     params["page_token"] = page_token

        response = requests.get(URL, params=params, headers=HEADERS)
        if response.status_code != 200: #если 200 то ОК
            print(f"ошибка получения данных {metric} coinmetrics: {response.status_code}")
            return

        res_json = response.json()
        records = res_json.get("data", [])
        all_records.extend(records)

        page_token = res_json.get("next_page_token")
        if not page_token:
            break #взяли все данные

    if not all_records:
        print("новых данных нет.")
        return

    new_df = pd.DataFrame(all_records).dropna(subset=[metric])
    new_df["date"] = pd.to_datetime(new_df["time"]).dt.strftime("%Y-%m-%d")
    new_df["value"] = pd.to_numeric(new_df[metric]).astype(int)
    new_df = new_df[["date", "value"]]

    final_df = pd.concat([existing_df, new_df]).drop_duplicates(
        subset=["date"], keep="last"
    )
    final_df = final_df.sort_values(by="date").reset_index(drop=True)

    final_df.to_csv(FILENAME, index=False)
    print(
        f"данные по {metric} обновлены в {FILENAME}. все прошло хорошо"
    )

#coin_metrics_get_csv("./data/csv/avg_size.csv", "BlkSizeByte") - это платная метрика к сожалению

print("\nПАРСИНГ СРЕДНИЙ РАЗМЕР ТРАНЗАЦИИ") #в байтах

url = "https://api.blockchain.info/charts/avg-block-size?timespan=all&sampled=false&format=csv" #BlockChain info - без API ключа
resp = requests.get(url)

if(resp.status_code == 200):
    df = pd.read_csv(
        url, header=None, names=["date", "value"], parse_dates=["date"]
    )
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    df["value"] = df["value"].astype(int)  # размер в байтах

    df.to_csv("./data/csv/avg_size.csv", index=False)
    print("данные по среднему размеру блока сохранены в ./data/csv/avf_size.csv. все прошло хорошо")
else:
    print(f"ошибка по сбору данных среднего размера транзакции с Blockchain.info: {resp.status_code}")

print("\nПРСИНГ СУТОЧНЫЙ ОБЪЕМ ПЕРЕВОДОВ")

#coin_metrics_get_csv("./data/csv/transfers_volume_sum.csv", "TxTfrValAdjUSD") #не работает

dfbtc = pd.read_csv("./data/csv/btc_no_vibrosi_copy.csv")

url = "https://api.blockchain.info/charts/estimated-transaction-volume-usd?timespan=all&sampled=false&format=csv"
resp = requests.get(url)

if(resp.status_code == 200):
    df = pd.read_csv(
        url, header=None, names=["date", "value"], parse_dates=["date"]
    )
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    #df["value"] = df["value"].astype(int)


    merged = pd.merge(df, dfbtc, on="date")

    merged["value_usd"] = merged["value"] * merged["close"] #нужно потому что мы раньше хранили в долларах, а он сохраняет в БТК

    df["value"] = merged["value_usd"]

    df.to_csv("./data/csv/transfers_volume_sum.csv", index=False)
    print("данные по объему переводов сохранен в ./data/csv/transfers_volume_sum.csv. Все прошло хорошо")
else:
    print(f"ошибка по сбору данных среднего размера транзакции с Blockchain.info: {resp.status_code}")

print("\nПРСИНГ volume_sum.csv с Blockchain.info")

url = "https://api.blockchain.info/charts/estimated-transaction-volume-usd?timespan=all&sampled=false&format=csv"
print("Загрузка объема транзакций (Blockchain.info)...")

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
response = requests.get(url, headers=headers)

if response.status_code == 200:
    with open("./data/csv/volume_sum_temp.csv", "wb") as f:
        f.write(response.content)
        
    # У Blockchain.info нет заголовков в CSV, проставляем их вручную
    df = pd.read_csv("./data/csv/volume_sum_temp.csv", names=['date', 'volume_sum'])
    df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
    
    df.to_csv("./data/csv/volume_sum.csv", index=False)
    print("Данные объема успешно сохранены в volume_sum.csv")
else:
    print(f"Ошибка получения объема: {response.status_code}")
