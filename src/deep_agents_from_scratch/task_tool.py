"""하위 에이전트를 통한 컨텍스트 격리를 위한 작업 위임 도구.

이 모듈은 격리된 컨텍스트를 가진 하위 에이전트를 생성하고 관리하기 위한 핵심 인프라를 제공합니다. 
하위 에이전트는 특정 작업 설명만 포함된 깨끗한 컨텍스트 창으로 작동함으로써 컨텍스트 충돌을 방지합니다.
"""

from typing import Annotated, NotRequired
from typing_extensions import TypedDict

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState, create_react_agent
from langgraph.types import Command

from deep_agents_from_scratch.prompts import TASK_DESCRIPTION_PREFIX
from deep_agents_from_scratch.state import DeepAgentState


class SubAgent(TypedDict):
    """특화된 하위 에이전트용 구성."""

    name: str
    description: str
    prompt: str
    tools: NotRequired[list[str]]


def _create_task_tool(tools, subagents: list[SubAgent], model, state_schema):
    """서브 에이전트를 통해 컨텍스트 격리를 가능하게 하는 작업 위임 도구를 생성합니다.

    이 기능은 격리된 컨텍스트를 가진 특수화된 서브 에이전트를 생성하는 핵심 패턴을 구현하여
    복잡한 다단계 작업에서 컨텍스트 충돌과 혼란을 방지합니다.

    Args:
        tools: 서브 에이전트에 할당 가능한 사용 가능한 도구 목록
        subagents: 특수화된 서브 에이전트 구성 목록
        model: 모든 에이전트에 사용할 언어 모델
        state_schema: 상태 스키마 (일반적으로 DeepAgentState)

    Returns:
        특수화된 서브 에이전트에 작업을 위임할 수 있는 ‘작업’ 도구
    """
    # Create agent registry
    agents = {}

    # Build tool name mapping for selective tool assignment
    tools_by_name = {}
    for tool_ in tools:
        if not isinstance(tool_, BaseTool):
            tool_ = tool(tool_)
        tools_by_name[tool_.name] = tool_

    # Create specialized sub-agents based on configurations
    for _agent in subagents:
        if "tools" in _agent:
            # Use specific tools if specified
            _tools = [tools_by_name[t] for t in _agent["tools"]]
        else:
            # Default to all tools
            _tools = tools
        agents[_agent["name"]] = create_react_agent(
            model, prompt=_agent["prompt"], tools=_tools, state_schema=state_schema
        )

    # Generate description of available sub-agents for the tool description
    other_agents_string = [
        f"- {_agent['name']}: {_agent['description']}" for _agent in subagents
    ]

    @tool(description=TASK_DESCRIPTION_PREFIX.format(other_agents=other_agents_string))
    def task(
        description: str,
        subagent_type: str,
        state: Annotated[DeepAgentState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ):
        """특정 작업을 독립된 컨텍스트를 가진 전문화된 하위 에이전트에게 위임합니다.

        이는 하위 에이전트에 작업 설명만 포함된 새로운 컨텍스트를 생성하여,
        상위 에이전트의 대화 기록으로 인한 컨텍스트 오염을 방지합니다.
        """
        # Validate requested agent type exists
        if subagent_type not in agents:
            return f"Error: invoked agent of type {subagent_type}, the only allowed types are {[f'`{k}`' for k in agents]}"

        # Get the requested sub-agent
        sub_agent = agents[subagent_type]

        # Create isolated context with only the task description
        # This is the key to context isolation - no parent history
        state["messages"] = [{"role": "user", "content": description}]

        # Execute the sub-agent in isolation
        result = sub_agent.invoke(state)

        # Return results to parent agent via Command state update
        return Command(
            update={
                "files": result.get("files", {}),  # Merge any file changes
                "messages": [
                    # Sub-agent result becomes a ToolMessage in parent context
                    ToolMessage(
                        result["messages"][-1].content, tool_call_id=tool_call_id
                    )
                ],
            }
        )

    return task
