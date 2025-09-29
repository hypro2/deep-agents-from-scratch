"""작업 계획 및 진행 상황 추적을 위한 TODO 관리 도구.

이 모듈은 구조화된 작업 목록을 생성하고 관리하는 도구를 제공하여
에이전트가 복잡한 워크플로를 계획하고 다단계 작업을 통해 진행 상황을 추적할 수 있도록 합니다.
"""

from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from deep_agents_from_scratch.prompts import WRITE_TODOS_DESCRIPTION
from deep_agents_from_scratch.state import DeepAgentState, Todo


@tool(description=WRITE_TODOS_DESCRIPTION,parse_docstring=True)
def write_todos(
    todos: list[Todo], tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    """작업 계획 및 추적을 위해 에이전트의 TODO 목록을 생성하거나 업데이트합니다.

    Args:
        todos: 내용과 상태가 포함된 할 일 항목 목록
        tool_call_id: 메시지 응답을 위한 도구 호출 식별자

    Returns:
        새로운 TODO 목록으로 에이전트 상태를 업데이트하는 명령어
    """
    return Command(
        update={
            "todos": todos,
            "messages": [
                ToolMessage(f"Updated todo list to {todos}", tool_call_id=tool_call_id)
            ],
        }
    )


@tool(parse_docstring=True)
def read_todos(
    state: Annotated[DeepAgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> str:
    """에이전트 상태에서 현재 TODO 목록을 읽습니다.

    이 도구를 통해 에이전트는 현재 TODO 목록을 조회하고 검토하여
    남은 작업에 집중하고 복잡한 워크플로우를 통해 진행 상황을 추적할 수 있습니다.

    Args:
        state: 현재 TODO 목록을 포함하는 주입된 에이전트 상태
        tool_call_id: 메시지 추적을 위한 주입된 도구 호출 식별자

    Returns:
        현재 TODO 목록의 형식화된 문자열 표현
    """
    todos = state.get("todos", [])
    if not todos:
        return "No todos currently in the list."

    result = "Current TODO List:\n"
    for i, todo in enumerate(todos, 1):
        status_emoji = {"pending": "⏳", "in_progress": "🔄", "completed": "✅"}
        emoji = status_emoji.get(todo["status"], "❓")
        result += f"{i}. {emoji} {todo['content']} ({todo['status']})\n"

    return result.strip()
