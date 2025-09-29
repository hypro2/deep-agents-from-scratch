"""TODO 추적 및 가상 파일 시스템을 지원하는 심층 에이전트용 상태 관리.

이 모듈은 다음을 지원하는 확장된 에이전트 상태 구조를 정의합니다:
- TODO 목록을 통한 작업 계획 및 진행 상황 추적
- 상태에 저장된 가상 파일 시스템을 통한 컨텍스트 오프로딩
- 리듀서 함수를 통한 효율적인 상태 병합
"""

from typing import Annotated, Literal, NotRequired
from typing_extensions import TypedDict

from langgraph.prebuilt.chat_agent_executor import AgentState


class Todo(TypedDict):
    """ 워크플로우를 통해 진행 상황을 추적하기 위한 구조화된 작업 항목.

    Attributes:
        content: 작업에 대한 간결하고 구체적인 설명
        status: Current state - pending, in_progress, or completed
    """

    content: str
    status: Literal["pending", "in_progress", "completed"]


def file_reducer(left, right):
    """두 파일 사전(dictionary)을 병합하며, 오른쪽이 우선합니다.

    에이전트 상태의 files 필드에 대한 리듀서 함수로 사용되며,
    가상 파일 시스템에 대한 증분 업데이트를 가능하게 합니다.

    Args:
        left: 왼쪽 사전 (기존 파일)
        right: 오른쪽 사전 (새/업데이트된 파일)

    Returns:
        오른쪽 값이 왼쪽 값을 덮어쓴 병합된 사전
    """
    if left is None:
        return right
    elif right is None:
        return left
    else:
        return {**left, **right}


class DeepAgentState(AgentState):
    """작업 추적 및 가상 파일 시스템을 포함하는 확장 에이전트 상태.

    LangGraph의 AgentState를 상속하며 다음을 추가합니다:
    - todos: 작업 계획 및 진행 상황 추적을 위한 할 일 항목 목록
    - files: 파일명을 콘텐츠에 매핑하는 딕셔너리로 저장된 가상 파일 시스템
    """

    todos: NotRequired[list[Todo]]
    files: Annotated[NotRequired[dict[str, str]], file_reducer]
