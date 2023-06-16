import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.remote.webelement import WebElement

# Setup Selenium
chrome_options = Options()
chrome_options.add_argument("--headless")  # Run Chrome in headless mode

# Set path to ChromeDriver executable
webdriver_service = Service('path/to/chromedriver')  # Replace with the actual path to chromedriver


driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

# URL of the paper
paper_url = 'https://inspirehep.net/literature/2658434'

driver.get(paper_url)

# figure_elements = driver.find_element(By.CLASS_NAME, "__Figure__ bg-white pa3")
try:
    elem = WebDriverWait(driver,10).until(
        EC.presence_of_element_located((By.CLASS_NAME, "ant-tabs-tab-btn"))
    )
    driver.find_elements(by=By.CLASS_NAME,value="ant-tabs-tab-btn")[1].click()
finally:
    pass

try:
    elem = WebDriverWait(driver,10).until(
        EC.presence_of_element_located((By.CLASS_NAME, "mv1"))
    )
finally:
    pass


def fetch_image_caption(element):
    text = element.text
    url = element.find_element(by=By.TAG_NAME,value="img").get_attribute("src")
    return text,url

gallery = []
elem = driver.find_elements(by=By.CLASS_NAME,value="mv1")
time.sleep(1)
try:
    def check_complete(driver):
        print("Page is ready!")
        return driver.execute_script('return document.readyState') == 'complete'
    WebDriverWait(driver, 10).until(check_complete)

finally:
    elem[0].click()
    figure_num = len(elem)

output = []
gallery = []
try:
    def check_complete(driver):
        print("Page is ready!")
        return driver.execute_script('return document.readyState') == 'complete'
    WebDriverWait(driver, 10).until(check_complete)
finally:
    # current_length = 0
    # have_new = True
    # time.sleep(1)
    # while have_new:
    #     time.sleep(1)
    #     gallery: WebElement = driver.find_elements(by=By.CLASS_NAME,value="ReactModalPortal")[1].find_elements(by=By.CLASS_NAME,value="mv1")
    #     current_length = len(gallery)
    #     print(gallery)
    #     if current_length == figure_num:
    #         have_new = False
    driver.find_elements(by=By.CLASS_NAME,value="ReactModalPortal")[1].find_elements(by=By.CLASS_NAME,value="anticon")[0].click()
    print("Start to fetch image caption")
    for i in range(figure_num):
        print(i)
        elem[i].click()
        time.sleep(1)        
        while len(gallery)<=i+1:
            gallery = driver.find_elements(by=By.CLASS_NAME,value="ReactModalPortal")[1].find_elements(by=By.CLASS_NAME,value="mv1")
            time.sleep(1)
            if len(gallery)>=i:
                figure_element = gallery[i]
                image_caption = fetch_image_caption(figure_element)
                print(image_caption)
                output.append(image_caption)
        time.sleep(2)
        driver.find_elements(by=By.CLASS_NAME,value="ReactModalPortal")[1].find_elements(by=By.CLASS_NAME,value="anticon")[0].click()
    