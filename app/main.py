import sys

from langchain.agents import AgentExecutor, Tool, create_react_agent
from langchain_community.tools import ShellTool
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

shell_tool = ShellTool()
tools = [
    Tool(
        name=shell_tool.name,
        description=shell_tool.description,
        func=shell_tool.run,
    )
]

template = '''Answer the following questions as best you can. You have access to the following tools:

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
Thought: {agent_scratchpad}'''
prompt = PromptTemplate.from_template(template)

agent = create_react_agent(
    ChatOpenAI(model="gpt-3.5-turbo"),
    tools,
    prompt,
)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

agent_executor.invoke({
    "input": sys.argv[1]
})
