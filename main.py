from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
load_dotenv()

from langchain import hub
from langchain.agents import AgentExecutor
from langchain.agents.react.agent import create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults 
from langchain_tavily import TavilySearch

tools = [
    TavilySearch(),
    # TavilySearchResults()
]
llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
react_prompt = hub.pull("hwchase17/react")
agent = create_react_agent(llm, tools, react_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
chain = agent_executor

def main():
    result = chain.invoke(
        input= {"input": "search for 3 job posting of lead ai engineer in the UAE and list their details and salary more than 30000 aed "}
    )
    print(result["output"])


if __name__ == "__main__":
    main()
