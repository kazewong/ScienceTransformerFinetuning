import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import TimeoutException
import requests

def search_papers(keyword, offset=0, limit=10):
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": keyword,
        "offset": offset,  # Specify the offset of the first paper to retrieve
        "limit": limit  # Specify the number of papers to retrieve
    }
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        papers_data = response.json()
        papers = papers_data.get('data', [])
        return papers
    else:
        raise Exception(f"Error: {response.status_code} - {response.text}")

directory = '/mnt/home/wwong/ceph/MLProject/Dataset/PaperScrapping/GW/'
batch_size = 100
total_number = 3000
keyword = "gravitational waves"

paper_id = []
for i in range(0,total_number,batch_size):
    retry_count = 0
    while retry_count < 5:
        try:
            papers = search_papers("gravitational waves", offset=i, limit=batch_size)
            print(f"Retrieved {len(papers)} papers with offset {i}")
            paper_id += [paper['paperId'] for paper in papers]
            time.sleep(5)  # Wait for 5 seconds before making another request
            break
        except:
            print(f"Error: retry {retry_count}")
            time.sleep(5)
            retry_count += 1

with open(directory+'paper_id.txt', 'w') as f:
    for item in paper_id:
        f.write("%s\n" % item)