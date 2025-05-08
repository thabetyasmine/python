from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser  



load_dotenv()

# llm = ChatGroq(
#     api_key=os.environ["GROQ_API_KEY"],  
#     model="llama3-70b-8192",
#     temperature=0.3,
#     max_tokens=500,
# )

# response = llm.invoke("What are the 7 wonders of the world?")
# print(f"Response: {response.content}")

# Initialize the Groq chat model
# chat_model = ChatGroq(
#       api_key=os.environ["GROQ_API_KEY"],  
#       model="llama3-70b-8192",
#       temperature=0.7,  # Slightly higher temperature for more creative responses
#       max_tokens=500,
# )

# Define the system message for pirate personality with emojis
# system_message = SystemMessage(
#       content="You are a friendly pirate who loves to share knowledge. Always respond in pirate speech, use pirate slang, and include plenty of nautical references. Add relevant emojis throughout your responses to make them more engaging. Arr! ☠️🏴‍☠️"
# )

# Define the question
# question = "What are the 7 wonders of the world?"

# Create messages with the system instruction and question
# messages = [
#       system_message,
#       HumanMessage(content=question)
# ]

# Get the response
# response = chat_model.invoke(messages)

# Print the response
# print("\nQuestion:", question)
# print("\nPirate Response:")
# print(response.content)

# Initialize the Groq chat model
# llm = ChatGroq(
#     api_key=os.environ["GROQ_API_KEY"],  
#     model="llama3-70b-8192",
#     temperature=0.3,
#     max_tokens=500,
# )

# The description helps the LLM to know what it should put in there.
# class Movie(BaseModel):
#     title: str = Field(description="The title of the movie.")
#     genre: list[str] = Field(description="The genre of the movie.")
#     year: int = Field(description="The year the movie was released.")

# parser = PydanticOutputParser(pydantic_object=Movie)

# Create a prompt template for generating movie titles
# movie_list_prompt_template = PromptTemplate.from_template(
#     "List {n} movie titles from the {category} category (name ,genre,year)."
# )

# # Create a runnable chain using the pipe operator
# movie_list_chain = movie_list_prompt_template | llm

# prompt_template_text = """
# rensponse with  no explanation o nothing just movie ,genre and year be consice:\n
# {format_instructions}\n
# {query}
# """

# format_instructions = parser.get_format_instructions()
# movie_prompt_template = PromptTemplate(
#     template=prompt_template_text,
#     input_variables=["query"],
#     partial_variables={"format_instructions": format_instructions},
# )

# prompt = movie_prompt_template.format(query="A 90s movie with Nicolas Cage.")
# text_output = llm.invoke(prompt)
# print(text_output.content)  

# parsed_output = parser.parse(text_output.content)
# print(parsed_output)    

# Run the chain with specific parameters
# response = movie_list_chain.invoke({
#     "n": 5,
#     "category": "Sci-Fi"
# })

# Print the response
# print("\nPrompt: List 5 movie titles from the Sci-Fi category (name,genre,year ).")
# print("\nResponse:")
# print(response.content)
# pip install langchain langchain_community langchain_groq duckduckgo-search

from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.chains import LLMMathChain
from langchain.prompts import PromptTemplate
from langchain.agents import AgentExecutor, Tool
from langchain.agents.structured_chat.base import StructuredChatAgent

# Initialize the Groq chat model
llm = ChatGroq(
    api_key=os.environ["GROQ_API_KEY"],  
    model="llama3-70b-8192",
    temperature=0.3,
    max_tokens=1024,
)

# Custom prompt for LLMMathChain
math_prompt = PromptTemplate.from_template(
    "Calculate the following expression and return the result in the format 'Answer: <number>': {question}"
)

# Set up the math chain
llm_math_chain = LLMMathChain.from_llm(llm=llm, prompt=math_prompt, verbose=True)

# Initialize tools
search = DuckDuckGoSearchRun()

# Wrap calculator with correct input handling
def calculator_func(question: str) -> str:
    return llm_math_chain.run(question)

calculator = Tool(
    name="calculator",
    description="Use this tool for arithmetic calculations. Input should be a mathematical expression.",
    func=calculator_func,
)

# List of tools for the agent
tools = [
    Tool(
        name="search",
        description="Search the internet for information about current events, data, or facts. Use this for looking up population numbers, statistics, or other factual information.",
        func=search.run
    ),
    calculator
]

# Create the agent using StructuredChatAgent
agent = StructuredChatAgent.from_llm_and_tools(
    llm=llm,
    tools=tools
)

# Initialize the agent executor
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    handle_parsing_errors=True
)

# Run the agent
result = agent_executor.invoke({"input": "What is the population difference between Tunisia and Algeria?"})
print(result["output"])


