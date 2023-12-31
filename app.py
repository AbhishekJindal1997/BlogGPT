import os
from dotenv import load_dotenv, find_dotenv
import requests
import streamlit as st
import re
import json
import http.client
from bs4 import BeautifulSoup
from termcolor import colored
import concurrent.futures
from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser, ZeroShotAgent
from langchain.utilities import GoogleSearchAPIWrapper, GoogleSerperAPIWrapper, ApifyWrapper, ArxivAPIWrapper
from langchain.utilities.wolfram_alpha import WolframAlphaAPIWrapper
from langchain.document_loaders.base import Document
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader, UnstructuredHTMLLoader
from langchain.text_splitter import TokenTextSplitter


load_dotenv()

st.set_page_config(page_title="News Article Generator :moneybag: :bulb: :computer: ",
                   page_icon=":computer:")

st.header('News Article Generator :newspaper: :bulb: :computer: v2.0.0')

# Vector Store
index = VectorstoreIndexCreator()

# Get Tools
googleSearch = GoogleSearchAPIWrapper()
googleSerp = GoogleSerperAPIWrapper()
# arxiv = ArxivAPIWrapper()
wolfram = WolframAlphaAPIWrapper()


# Load Tools
tools = [
    Tool(
        name="Google Search",
        description="Useful for when you need to answer questions about current events",
        func=googleSearch.run
    ),
    Tool(name="Wolfram Alpha",
         description="Ideal for mathematical problems, statistics, data analysis, and fact-based queries.",
         func=wolfram.run
         ),
    Tool(name="Google SERP",
         description=" Ideal for analyzing search engine result patterns, SEO research, and finding specific types of search result data.",
         func=googleSerp.run),

]


# Load Memory
memory = ConversationBufferMemory(memory_key="chat_history")

# Load Model
llm = ChatOpenAI(
    temperature=0.8,
    model="gpt-3.5-turbo-16k",
)

# query = "Manitoba RCMP say transport truck in crash that killed 15 had right-of-way"


# 1/ Search the web for articles/blogs related to the input
def serp_search(query):

    conn = http.client.HTTPSConnection("google.serper.dev")
    payload = json.dumps({
        "q": query,
    })
    headers = {
        'X-API-KEY': os.getenv('SERPER_API_KEY'),
        'Content-Type': 'application/json'
    }
    conn.request("POST", "/search", payload, headers)
    res = conn.getresponse()
    res_data = res.read()
    # print('Search Results', res_data.decode("utf-8"))
    print(colored("Found Articles for the input provided ", 'green'))
    return res_data.decode("utf-8")


# 2/ Given the search results, find the best n articles URL's
def find_best_article_urls(res_data, query):

    res_str = json.dumps(res_data)

    template = """ 
    You are a world class journalist and researcher, you are extremely good at finding most 
    relevant articles for certain topics: 
    {res_str}
    Above is the list of search results for the query: {query}
    Please choose the best 8 articles from the list above, return only an array of url's, 
    do not include anything else: return ONLY an array of url's, nothing else.
    """

    # Create the prompt
    prompt_template = PromptTemplate(
        input_variables=["res_str", "query"], template=template)

    article_picker_chain = LLMChain(
        llm=llm, prompt=prompt_template, verbose=False)

    urls = article_picker_chain.predict(res_str=res_str, query=query)

    # Convert String to list
    url_list = json.loads(urls)
    print(colored('Found URL"s for the articles', 'green'))
    return url_list


# 3/ Get Page Data from UR's
def get_page_data_from_urls(urls):

    data = []

    for url in urls:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        text = ' '.join(paragraph.text for paragraph in paragraphs)
        text = text.replace('\n', '').replace('\t', '')
        # append the text from each url to an array
        data.append(text)

    # Save the data to a single markdown file
    filename = 'scraped_data' + '.md'
    with open(filename, 'w',  encoding='utf-8') as f:
        f.write('\n'.join(data))

    print(colored("Found Data from URL's", "green"))
    print(colored('Data has been saved successfully in a markdown file', 'green'))
    # print(colored(data, "green"))
    return data

# 4/ Summarize the data into a blog or article
def summarize(data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=200,
        length_function=len)
    
    text = text_splitter.create_documents(data)

    
    template = """
        {text}
        You are a world class journalist, You will try to summarize the text.

        Please follow all of the following rules:

        1/ Make sure the content is engaging, infromative with good data
        2/ Make sure the content is original and not plagiarized
        4/ Make sure the content is not too short, keep it long enough to be informative
        5/ Make sure the content is not too boring, keep it interesting and engaging
        6/ The content need to give audience actionable adive and insights too
    """

    prompt_template = PromptTemplate(
        input_variables=["text"], template=template)

    summarizer_chain = LLMChain(llm=llm, prompt=prompt_template, verbose=False)

    summaries = []

    # Synchronous Processing: This involves processing one chunk at a time.
    for chunk in enumerate(text):
        summary = summarizer_chain.predict(text=chunk)
        summaries.append(summary)

    # Asynchronous Processing: This involves processing multiple chunks at the same time.
    # with concurrent.futures.ThreadPoolExecutor() as executor:
    #     futures = [executor.submit(summarizer_chain.predict, text=chunk) for chunk in text]
    #     for future in concurrent.futures.as_completed(futures):
    #         summaries.append(future.result())


    print(colored('Data Summarized successfully', 'green'))

    # Save the data to a single markdown file
    filename = 'Summary' + '.md'
    with open(filename, 'w',  encoding='utf-8') as f:
        f.write('\n'.join(summaries))
    return summaries



# 5/ Create a blog post or article
def create_blog_post(summaries, urls):
    template = """
    {summaries}
    You are a world-class news journalist and your task is to create a highly detailed and informative news article, 
    incorporating specific timeframes of the incidents.

    Use all the relevant information from the summaries and follow the guidelines below:

    1/ Your primary objective is to compose an extensive news report, maintaining a formal and unbiased tone throughout
      the article.
    2/ Be sure to use transition words to enhance the flow and coherence of the text
    3/ The article should exceed 1,000 words to provide a thorough analysis of the subject. 
    4/ Create a succinct and engaging headline for the article, along with a subheadline that effectively 
       summarizes the key points of the story
    5/ The piece should feature an engaging introduction.
    6/ Craft a concise meta description maximum of 160 characters, which succinctly encapsulates the essence of the article.
    7/ Add a fact check / references paragrpah using the following urls as soruces {urls}, provide url links to the users.
    8/ Include relevant date and timeframes of the incident, wherever relevant to make the article more informative.
    9/ Provide a focus keyphrase for the article, which should be used in the first paragraph of 
       the article, in meta description, and in the headline.

    Your writing should guide the reader through the incident, explaining each element clearly and engagingly, also utlize
    date and timeframes to make the article more informative. Remember, your role as a journalist is to make even complex
    situations easily understandable to your audience.
    """

    prompt_template = PromptTemplate(
        input_variables=["summaries", "urls"], template=template)

    create_blog_chain = LLMChain(
        llm=llm, prompt=prompt_template, verbose=False)

    blog_post = create_blog_chain.predict(
        summaries=summaries, urls=urls)

    # Save the data to a single markdown file
    filename = 'Blog' + '.md'
    with open(filename, 'w',  encoding='utf-8') as f:
        f.write(blog_post)

    # print(blog_post)
    print(colored('Blog Post Created Successfully', 'green'))
    return blog_post


# START
query = st.sidebar.text_input("Enter a Blog Topic: ")

if query:
    # Progress bar
    progress = st.progress(0)
    progress.text("Starting...")

    # Step 1/
    progress.text("Step 1/ Search the web for articles/blogs related to the input, In progress ⏳")
    print(colored("Step 1/ Search the web for articles/blogs related to the input", 'blue'))
    res_data = serp_search(query)
    if res_data:
        st.sidebar.markdown("1/ Find Relevant Articles ✅")
        progress.text("Step 1 completed...")
        progress.progress(20)

    # Step 2/
    progress.text("Step 2/ Finding the best articles URL's, In progress ⏳")
    print(colored("Step 2/ Given the search results, find the best articles URL's", "blue"))
    urls = find_best_article_urls(res_data, query)
    if urls:
        st.sidebar.markdown("2/ Extract Articles URL's ✅")
        progress.text("Step 2 completed...")
        progress.progress(40)

    # Step 3/
    progress.text("Step 3/ Extracting Data from URL's, In progress ⏳")
    print(colored("Step 3/ Get Page Data from URL's", "blue"))
    data = get_page_data_from_urls(urls)
    if data:
        st.sidebar.markdown("3/ Extracted Data from URL's ✅")
        progress.text("Step 3 completed...")
        progress.progress(60)

    # Step 4/
    progress.text("Step 4/ Summarizing the Extracted Data, In progress ⏳")
    print(colored("Step 4/ Summarize the data", "blue"))
    summaries = summarize(data)
    if summaries:
        st.sidebar.markdown("4/ Summarize the Extracted Data ✅")
        progress.text("Step 4 completed...")
        progress.progress(80)

    # Step 5/
    progress.text("Step 5/ Creating the Final Blog Post, Please Wait... ⏳")
    print(colored("Step 5/ Create a blog post or article", "blue"))
    blog_post = create_blog_post(summaries, urls)
    if blog_post:
        st.sidebar.markdown("5/ Final Blog Post ✅")
        st.markdown(f"\n{blog_post}")
        progress.text("Step 5 completed...")
        progress.progress(100)
