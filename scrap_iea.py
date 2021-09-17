import selenium
from selenium import webdriver
import time
import json
import os
if __name__ == "__main__":
    driver = webdriver.Firefox()
    country = "UK"
    years = [str(1990 + k) for k in range(30)]
    data = {country: {}}
    for year in years:
        driver.get(
            "https://www.iea.org/data-and-statistics/data-tables?country={country}&energy=Balances&year={year}".format(
                country=country, year=year))
        keys = []

        time.sleep(2)
        for k in range(1, 13):
            element = driver.find_elements_by_xpath(
                "/html/body/div[6]/div/main/div[2]/div[2]/div/div/div/div[2]/div/div/table/thead/tr/th[{k}]/span".format(
                    k=k))
            for el in element:
                keys += [el.text]

        data[country].update({year: {key: {} for key in keys}})
        print(data)
        for j in range(2, len(keys) + 2):
            for k in range(2, 30):
                current_key = driver.find_elements_by_xpath(
                    "/html/body/div[6]/div/main/div[2]/div[2]/div/div/div/div[2]/div/div/table/tr[{k}]/td[1]".format(
                        k=k))[0].text
                element = driver.find_elements_by_xpath(
                    "/html/body/div[6]/div/main/div[2]/div[2]/div/div/div/div[2]/div/div/table/tr[{k}]/td[{j}]".format(
                        j=j, k=k))
                for el in element:
                    value = el.text.replace("\u202f", "")
                    if value != "":
                        data[country][year][keys[j - 2]].update({current_key: float(el.text.replace("\u202f", ""))})
                    else:
                        data[country][year][keys[j - 2]].update({current_key: 0})
    if os.path.exists("data/iea/data.json"):
        with open("data/iea/data.json", "r") as json_file:
            old_data = json.load(json_file)
    data.update(old_data)
    with open("data/iea/data.json", "w") as json_file:
        json.dump(data, json_file)
