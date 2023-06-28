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
import json
import os
import urllib.request

def download_file(url, save_as):
    urllib.request.urlretrieve(url, save_as)


def get_paper(paper_id, get_references=False):
    url = f"https://api.semanticscholar.org/v1/paper/{paper_id}"
    response = requests.get(url)
    
    if response.status_code == 200:
        paper_data = response.json()

        arxivId = paper_data.get('arxivId', None)
        is_open_access = paper_data.get('is_open_access', False)
        paperId = paper_data.get('paperId', None)
        title = paper_data.get('title', None)
        if get_references==True:
            references = paper_data.get('references', [])
            return {"paperId":paperId,"arxivId":arxivId,"is_open_access":is_open_access,"title":title,"references":references}
        else:
            return {"paperId":paperId,"arxivId":arxivId,"is_open_access":is_open_access,"title":title}

    else:
        print(f"Error: {response.status_code} - {response.text}")
        return []

# Setup Selenium
chrome_options = Options()
chrome_options.add_argument("--headless")  # Run Chrome in headless mode

# Set path to ChromeDriver executable
webdriver_service = Service('path/to/chromedriver')  # Replace with the actual path to chromedriver
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

# URL of the paper

directory = '/mnt/home/wwong/ceph/MLProject/Dataset/PaperScrapping/GW/'
tag = '0'
paper_id = '30d50f507f9f8a4ee22546daecf8d179387dac01'

def download_paper(directory, tag, paper_id):
    metadata = get_paper(paper_id)
    driver.get('https://www.semanticscholar.org/paper/'+paper_id)
    try:
        WebDriverWait(driver,5).until(EC.presence_of_element_located((By.CLASS_NAME,"figure-list__figure")))
        elem_list = driver.find_elements(by=By.CLASS_NAME,value="figure-list__figure")
        image_caption = []
        elem_list[0].click()
        open_flag = False
        for i in range(len(elem_list)):
            print(i)
            try:
                if open_flag ==False:
                    driver.find_element(by=By.CLASS_NAME,value="modal__with-footer").find_element(by=By.CLASS_NAME,value='cl-button__label').click()
                    open_flag = True
            except Exception as e:
                print(e)
            caption = driver.find_element(by=By.CLASS_NAME,value="modal__with-footer").find_element(by=By.CSS_SELECTOR,value='span').text
            image_link = driver.find_element(by=By.CLASS_NAME,value="modal__with-footer").find_element(by=By.CSS_SELECTOR,value='img').get_attribute('src')
            image_caption.append({"caption":caption,"image_link":image_link})
            driver.find_element(by=By.CLASS_NAME,value="next-figure").click()
            time.sleep(0.1)
        metadata['num_figures'] = len(elem_list)
        output = {"metadata":metadata,"image_caption":image_caption}
        if not os.path.exists(directory+tag):
            os.makedirs(directory+tag)
        if not os.path.exists(directory+tag+"/"+metadata['arxivId']):
            os.makedirs(directory+tag+"/"+metadata['arxivId'])
        with open(directory+tag+"/"+metadata['arxivId']+"/metadata.json",'w') as file:
            json.dump(output, file, indent=4)
        
        for i in image_caption:
            download_file(i['image_link'],directory+tag+"/"+metadata['arxivId']+"/"+i['image_link'].split("/")[-1])

        return True
    except TimeoutException:
        print("Loading took too much time! Could be a paper without figures")
        return False
    except Exception:
        return False


download_paper(directory, tag, paper_id)