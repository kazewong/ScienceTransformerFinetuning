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
paper_url = 'https://www.semanticscholar.org/paper/Constraining-gravitational-wave-amplitude-with-Ng-Isi/30d50f507f9f8a4ee22546daecf8d179387dac01'

driver.get(paper_url)


elem_list = driver.find_elements(by=By.CLASS_NAME,value="figure-list__figure")
image_caption = []
elem_list[0].click()
for i in range(len(elem_list)):
    print(i)
    try:
        driver.find_element(by=By.CLASS_NAME,value="modal__with-footer").find_element(by=By.CLASS_NAME,value='cl-button__label').click()
        time.sleep(0.2)
    except:
        pass
    caption = driver.find_element(by=By.CLASS_NAME,value="modal__with-footer").find_element(by=By.CSS_SELECTOR,value='span').text
    image_link = driver.find_element(by=By.CLASS_NAME,value="modal__with-footer").find_element(by=By.CSS_SELECTOR,value='img').get_attribute('src')
    image_caption.append((caption,image_link))
    driver.find_element(by=By.CLASS_NAME,value="next-figure").click()
    time.sleep(0.1)