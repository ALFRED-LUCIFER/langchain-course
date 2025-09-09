from typing import List, Union
from dotenv import load_dotenv
from langchain.agents import tool
from langchain_core.agents import AgentAction, AgentFinish
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.tools.render import render_text_description
from langchain_core.tools import BaseTool

load_dotenv()


@tool
def get_text_length(text: str) -> int:
    """Returns the length of the input text by characters"""
    print(f"Calling get_text_length enter with {text=}")
    text = text.strip("'\n").strip(
        '"'
    )  # stripping away non alpahbetic characters just incase
    return len(text)


def find_tool_by_name(tools: List[BaseTool], tool_name: str) -> BaseTool:
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise ValueError(f"Tool with name {tool_name} not found")

def main():
    print("Hello from langchain-course!")
    tools = [get_text_length]

    template = """
    Answer the following questions as best you can. You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!

    Question: {input}
    Thought:

    """
    prompt = PromptTemplate.from_template(template=template).partial(
        tools=render_text_description(tools),
        tool_names=", ".join([tool.name for tool in tools]),
    )
    llm = ChatOpenAI(temperature=0, stop=["\nObservation"], model="gpt-4o-mini")
    agent = {"input": lambda x: x["input"]} | prompt | llm | ReActSingleInputOutputParser()
    agent_step: Union[AgentAction,AgentFinish] = agent.invoke(
        {"input": 'What is the text length of "Hello, world!"?| in characters?'}
    )
    if isinstance(agent_step, AgentAction):
        tool_name = agent_step.tool
        tool_to_use = find_tool_by_name(tools, tool_name)
        tool_input = agent_step.tool_input
        observation = tool_to_use.func(str(tool_input))
        print(f"{observation =}")
  



if __name__ == "__main__":
    main()
