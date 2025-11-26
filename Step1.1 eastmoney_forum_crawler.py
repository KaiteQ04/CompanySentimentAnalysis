import time
import requests
from bs4 import BeautifulSoup
import random
import re
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import openpyxl

#设置最小开始页数和最大页数
start_page=1
end_page=100000
#设置excel名字
excel_name='873726.xlsx'

#根据具体股票的url修改下列值。此url是该股票第一页的url
root_url="https://guba.eastmoney.com/list,zssh000001.html"
#新建excel表并保存
workbook=openpyxl.Workbook()
sheet=workbook.active
sheet['A1']="Date"
sheet['B1']="Title"
sheet['C1']="Article"
sheet['D1']="Read"
sheet['E1']="Like"
workbook.save(excel_name)

#伪造userAgent防止被封
def get_headers():
    user_agents=["Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
                 "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 Edg/124.0.0.0"]
    headers={
        "User-Agent":random.choice(user_agents),
        "Cookie":"qgqp_b_id=81f4db3c36355428f13b3cf1d277ed53; websitepoptg_api_time=1714399267652; st_si=36263058637056; cdcfh=5605094204491490%2C1876113287206580%2C2047345826970400%2C8781112044209498; p_origin=https%3A%2F%2Fpassport2.eastmoney.com; mtp=1; ct=CF4nnjAINopq9EJw9JJsHLdtIvLngJKc73zoyTzWtEplmcanyt2nC0UmjxMULc3R3UMkfqAfLRT28mbb-iNiOWxUlUYOaAplWcgDwGvTVi7i6HHhsMJr7s5cFtGUXRgsiGmT2s0nOQAx19oTip71MJ0qkqCuym8sxAexYHHz6Rc; ut=FobyicMgeV7IXKwLQXToga00suL68_EMZ7j6FdQfSw2slT1UsSPABQ9qf6Be9G32JSgwLj6vBZphKwXA9jpTu6XqIcP7qXc4ityaKbZJR166ICI1uAwfZHLIE0qXFdf0nzriHiXijNKRLm08qDPk4YR2LLHshgUIBJeK_ZrwZKBvPnihgLYketlGyShLauVuEFl-fKa-j0tzjn3T4JDJly89-n-ZoIexrXQRLqaO9XaDxQ2Blbm-mns4AtCgFfBYZIAIrlmjOqCLYZmjIdFL0xC_a7XE941DCqHGNLwBM8XSvbBWxdHDmc3L-ElCL5hBn_t6JEz4-tTwsjrHbR34rRH3oi6Rz4av; pi=4608067133655308%3Bt4608067133655308%3B%E8%82%A1%E5%8F%8BC6R2525909%3B8ZHR395Ea9K4%2FavF9iS4nNAA2FzHOpyA7CZz21tCejePI6N%2BXmU2knF81dpe4Rd8ZpRh6b7Rh7vzek5hPI%2Fwk7zD%2Bx9d908qYlmrn20gGrDzJ%2Bm8IZMuli3lUnoWY7HulQKTZkaGmX9SNPwBlyhPlvFkME8SqRdnZcuxzC3KM4%2BWkqloVoshy71ng1yHO%2F46keZmB7ko%3BOe3FRys%2FMycxy7PJ%2FZo976Bp0d%2FKCof8YikSnxjzWATVfbHf72A9be%2BfBKzZWK%2ByHWiFEphl%2BugiUNi22rHw%2BuS1L60WN3GHHufUgn6L%2BwGVedZg%2FZeVvlD03nX1ALbN1s5MZkLTk3zMO3F%2FYlzEeOpnhLpmFQ%3D%3D; uidal=4608067133655308%e8%82%a1%e5%8f%8bC6R2525909; sid=; vtpst=|; st_asi=delete; guba_blackUserList=; st_pvi=87234779525032; st_sp=2024-04-29%2022%3A01%3A07; st_inirUrl=https%3A%2F%2Fwww.baidu.com%2Flink; st_sn=48; st_psi=20240430093144520-119101302791-0094553225"
    }
    return headers

#判断url类型以及是否合法
def JudgeInitPattern(url):
    pattern1=r'\S*//caifuhao.eastmoney.com/news/\d+'
    pattern2=r'\S*/news,\w+,\w+.html'
    #pattern3=r'/news,\d+,\d+.html'
    if re.match(pattern1, url):
        return 1
    elif re.match(pattern2, url):
        return 2
    else:
        return 3

#判断日期是否在2014之后。返回1即代表不合法
def JudgeDatePattern(date):
    pattern1=r'\s*2013\S+'
    pattern2 = r'\s*2012\S+'
    pattern3 = r'\s*2011\S+'
    pattern4 = r'\s*2010\S+'
    if re.match(pattern1, date):
        return 1
    elif re.match(pattern2, date):
        return 1
    elif re.match(pattern3, date):
        return 1
    elif re.match(pattern4, date):
        return 1
    else:
        return 2

#获取单个帖子的正文、评论、点赞数
def get_single_ArticleLikeComment(url):
    if url==None:
        return

    #第一类评论url
    if JudgeInitPattern(url) == 1:
        # 使用Selenium打开浏览器
        browser = webdriver.Chrome()
        browser.get(url)
        #获取源码
        content = browser.page_source
        soup = BeautifulSoup(content, "html.parser")
        # 获取正文内容
        sentences = soup.find("div","article-body").find_all("p")
        article = ""
        #当不存在p时
        if len(sentences)==0:
            print("该页加载出错")
            return "wrong article","","","","","","","","","","",""

        flag=soup.find("p",class_="txt")
        for sentence in sentences:
            if sentence==flag:
                continue
            article = article + sentence.get_text() + "\n"
        #获取正文点赞数
        like=soup.find("span",class_="zancout text-primary").get_text()

        #获取5个热门评论及对应点赞数
        comments=soup.find_all("div",class_="level1_item clearfix")

        if 0 in range(len(comments)):
            comment1 = comments[0].find("div", class_="full_text").get_text()
            like1 = comments[0].find("span", class_="z_num").get_text()
        else:
            comment1 = ""
            like1 = ""

        if 1 in range(len(comments)):
            comment2 = comments[1].find("div", class_="full_text").get_text()
            like2 = comments[1].find("span", class_="z_num").get_text()
        else:
            comment2 = ""
            like2 = ""

        if 2 in range(len(comments)):
            comment3 = comments[2].find("div", class_="full_text").get_text()
            like3 = comments[2].find("span", class_="z_num").get_text()
        else:
            comment3 = ""
            like3 = ""

        if 3 in range(len(comments)):
            comment4 = comments[3].find("div", class_="full_text").get_text()
            like4 = comments[3].find("span", class_="z_num").get_text()
        else:
            comment4 = ""
            like4 = ""

        if 4 in range(len(comments)):
            comment5 = comments[4].find("div", class_="full_text").get_text()
            like5 = comments[4].find("span", class_="z_num").get_text()
        else:
            comment5 = ""
            like5 = ""

        #获取日期
        date=soup.find("span",class_="txt").get_text()

        #返回模式1下爬取到的正文、正文点赞数、评论及点赞数
        return article,like,comment1,like1,comment2,like2,comment3,like3,comment4,like4,comment5,like5,date

    #第二类url
    elif JudgeInitPattern(url)==2:
        # 使用Selenium打开浏览器
        browser = webdriver.Chrome()
        browser.get(url)
        #刷新网页并等待出现评论后，再获取源码
        browser.refresh()
        wait = WebDriverWait(browser, 60)
        element = wait.until(EC.presence_of_element_located((By.CLASS_NAME, 'likemodule')))
        #获取网页源码
        content = browser.page_source
        soup = BeautifulSoup(content, "html.parser")
        #记录正文内容
        sentences = soup.find_all("p")
        article = ""
        for sentence in sentences:
            article = article + sentence.get_text() + "\n"
        #记录正文点赞数
        like=soup.find("li",class_="likebox").get_text()

        #记录评论和点赞数
        comments=soup.find_all("div",class_="reply_item cl")

        if 0 in range(len(comments)):
            comment1 = comments[0].find("span",class_="reply_title_span").get_text()
            like1 = comments[0].find("span", class_="likemodule").get_text()
        else:
            comment1 = ""
            like1 = ""

        if 1 in range(len(comments)):
            comment2 = comments[1].find("span",class_="reply_title_span").get_text()
            like2 = comments[1].find("span", class_="likemodule").get_text()
        else:
            comment2 = ""
            like2 = ""

        if 2 in range(len(comments)):
            comment3 = comments[2].find("span",class_="reply_title_span").get_text()
            like3 = comments[2].find("span", class_="likemodule").get_text()
        else:
            comment3 = ""
            like3 = ""

        if 3 in range(len(comments)):
            comment4 = comments[3].find("span",class_="reply_title_span").get_text()
            like4 = comments[3].find("span", class_="likemodule").get_text()
        else:
            comment4 = ""
            like4 = ""

        if 4 in range(len(comments)):
            comment5 = comments[4].find("span",class_="reply_title_span").get_text()
            like5 = comments[4].find("span", class_="likemodule").get_text()
        else:
            comment5 = ""
            like5 = ""

        #获取日期
        date=soup.find("div",class_="time").get_text()

        return article, like, comment1, like1, comment2, like2, comment3, like3, comment4, like4, comment5, like5,date

    #错误
    else:
        print("错误的url")
        raise Exception("error")


#获取单个页面的所有帖子的信息
def pares_single_html(html):
    soup=BeautifulSoup(html,"html.parser")
    #print(soup)
    article_items=soup.find_all("tr",class_="listitem")
    count=0
    # 当页面的帖子条目为0，说明网页不存在，终止循环并返回1
    if len(article_items) == 0:
        return 2
    for article_item in article_items:
        count=count+1
        if not(count in range(1,81,20)):
            continue
        #获取目录中就能看到的信息
        read=article_item.find("div",class_="read").get_text()
        reply=article_item.find("div",class_="reply").get_text()
        title=article_item.find("div",class_="title").find("a").get_text()
        href=article_item.find("a")["href"]
        author=article_item.find("div",class_="author").find("a").get_text()
        #date=article_item.find("div",class_="update").get_text()

        #判断并修改url
        if JudgeInitPattern(href)==1:
            href="https:"+href
        elif JudgeInitPattern(href)==2:
            href="https://guba.eastmoney.com"+href
        print("爬取第",count,"条评论",href)

        #获取进入评论页面才能获得的信息
        article, like, comment1, like1, comment2, like2, comment3, like3, comment4, like4, comment5, like5,date = get_single_ArticleLikeComment(href)

        #如果日期在2014年之前，并跳出循环并返回2
        if JudgeDatePattern(date)==1:
            return 1

        wb = openpyxl.load_workbook(excel_name)
        sheet1=wb.active
        sheet1.append([date,title,article,read,like])
        wb.save(excel_name)

page_indexs=range(start_page,end_page,1)
count=start_page-1
#对所有存在的页获取源码并解析
for idx in page_indexs:
    count=count+1
    url = f"https://guba.eastmoney.com/list,873726_{idx}.html"
    r = requests.get(url, headers=get_headers())
    #若未获取页面的源码，报错
    if r.status_code != 200:
        raise Exception("error")
    html=r.text

    #开始解析单个页面
    print("第", count, "页爬取开始")
    flag=pares_single_html(html)
    if flag==1:
        print("爬取到2014年之前评论，爬取结束")
        break
    elif flag==2:
        print("该页面不存在，爬取结束")
        break
    else:
        print("第", count, "页爬取结束\n")

