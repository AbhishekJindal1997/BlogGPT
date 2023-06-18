import os
from dotenv import load_dotenv, find_dotenv
import requests
import streamlit as st
import re
import json
import http.client
from bs4 import BeautifulSoup
from termcolor import colored
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


load_dotenv()

st.set_page_config(page_title="Blog Post Generator :moneybag: :bulb: :computer:",
                   page_icon=":computer:")

st.header('Blog Post Generator :moneybag: :bulb: :computer:')

# Vector Store
index = VectorstoreIndexCreator()

# Get Tools
googleSearch = GoogleSearchAPIWrapper()
googleSerp = GoogleSerperAPIWrapper()
arxiv = ArxivAPIWrapper()
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


# 2/ Given the search results, find the best 5 articles URL's
def find_best_article_urls(res_data, query):

    res_str = json.dumps(res_data)

    template = """ 
    You are a world class journalist and researcher, you are extremely good at finding most 
    relevant articles for certain topics: 
    {res_str}
    Above is the list of search results for the query: {query}
    Please choose the best 7 articles from the list above, return only an array of url's, 
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
        chunk_size=8000,
        chunk_overlap=200,
        length_function=len)
    text = text_splitter.create_documents(data)

    template = """
    {text}
    You are a world class journalist, You will try to summarize the text within 1000 words

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

    for chunk in enumerate(text):
        summary = summarizer_chain.predict(text=chunk)
        summaries.append(summary)

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
        You are a nicheGPT your job is to create a blog / article, based on the following rules,

        Please follow all of the following rules:
        - Craft an entirely original, imaginative, and conversational-style, persuasive tone 
        article
        - Provide detailed, user engaging content and long-form content, enough to be informative.
        - Provide a clear and concise introduction, body, and conclusion.
        - Provide user with actionable advice and insights.
        - Provide a focus keyphrase, 
        - Include dates and timeframes wherever relevant.
        - Include contractions, idioms, transition words, interjections, dangling modifiers, 
           and colloquial language, while steering clear of redundancy and awkward phrasing.
        - The piece should feature a main SEO meta-title no more than 60 characters
        - Meta-description no more than 160 characters, slug, and excerpt in the introduction.
        - Incorporate bulleted or numbered lists as necessary.
        - Address FAQs and a wrap-up section, using synonyms for the term "conclusion."
        - Fact check your information by inserting citations and the links {urls} to the information to support your 
           statements.
        - Remember to use question marks for questions.
        - Avoid altering the original keyword or title when crafting the title.
        - Include the keyphrase or title 2-3 times within the text, and incorporate it into 
          headings too.

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
query = st.text_input("Enter a Blog Topic: ")

if query:

    # Step 1/
    print(colored("Step 1/ Search the web for articles/blogs related to the input", 'blue'))
    res_data = serp_search(query)

    with st.expander("1/ Find Relevant Articles"):
        st.info(res_data)

    # Step 2/
    print(colored("Step 2/ Given the search results, find the best articles URL's", "blue"))
    urls = find_best_article_urls(res_data, query)

    with st.expander("2/ Extract Articles URL's"):
        st.info(urls)

    # Step 3/
    print(colored("Step 3/ Get Page Data from URL's", "blue"))
    data = get_page_data_from_urls(urls)

    with st.expander("3/ Extracted Data from URL's"):
        st.info(data)

    # Step 4/
    print(colored("Step 4/ Summarize the data", "blue"))
    summaries = summarize(data)

    with st.expander("4/ Summarize the Extracted Data"):
        st.info(summaries)

    # Step 5/
    print(colored("Step 5/ Create a blog post or article", "blue"))
    blog_post = create_blog_post(summaries, urls)

    with st.expander("5/ Final Blog Post"):
        st.info(blog_post)
