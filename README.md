# 🧱 처음부터 만드는 딥 에이전트

<img width="720" height="289" alt="Screenshot 2025-08-12 at 2 13 54 PM" src="https://github.com/user-attachments/assets/90e5a7a3-7e88-4cbe-98f6-5b2581c94036" />

[Deep Research](https://academy.langchain.com/courses/deep-research-with-langgraph)는 코딩과 함께 주요 에이전트 활용 사례 중 하나로 부상했습니다. 이제 우리는 다양한 작업에 사용할 수 있는 범용 에이전트의 등장을 목격하고 있습니다. 예를 들어, [Manus](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus)는 장기적인 작업을 위한 에이전트로 상당한 주목과 인기를 얻었으며, 평균적인 Manus 작업은 약 50개의 도구 호출(tool call)을 사용합니다! 두 번째 예로, Claude Code는 코딩 이외의 작업에도 일반적으로 사용되고 있습니다. 이러한 인기 있는 "딥" 에이전트들 전반에 걸친 [컨텍스트 엔지니어링 패턴](https://docs.google.com/presentation/d/16aaXLu40GugY-kOpqDU4e-S0hD1FmHcNyF0rRRnb1OU/edit?slide=id.p#slide=id.p)을 자세히 살펴보면 몇 가지 공통된 접근 방식을 발견할 수 있습니다:

* **작업 계획 (예: TODO), 종종 recitation과 함께 사용**
* **파일 시스템으로 컨텍스트 오프로딩**
* **하위 에이전트(sub-agent) 위임을 통한 컨텍스트 격리**

이 과정에서는 LangGraph를 사용하여 이러한 패턴을 처음부터 구현하는 방법을 보여줍니다!

**한글 번역 hypro2:**

해당 번역본은 2025년 09월까지의 버전입니다.

example폴더에 deepagent에 있는 예제를 같이 보기 좋도록 추가했습니다.

## 🚀 빠른 시작

### 사전 준비

- Python 3.11 이상 버전을 사용하고 있는지 확인하세요.
- 이 버전은 LangGraph와의 최적의 호환성을 위해 필요합니다.
```bash
python3 --version
````

  - [uv](https://docs.astral.sh/uv/) 패키지 관리자

<!-- end list -->

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# Update PATH to use the new uv version
export PATH="/Users/$USER/.local/bin:$PATH"
```

### 설치

1.  리포지토리 복제:

<!-- end list -->

```bash
git clone https://github.com/langchain-ai/deep_agents_from_scratch
cd deep_agents_from_scratch
```

2.  패키지 및 의존성 설치 (가상 환경이 자동으로 생성 및 관리됩니다):

<!-- end list -->

```bash
uv sync
```

3.  프로젝트 루트에 API 키를 담을 `.env` 파일을 생성합니다:

<!-- end list -->

```bash
# .env 파일 생성
touch .env
```

`.env` 파일에 API 키를 추가하세요:

```env
# 외부 검색 기능이 있는 리서치 에이전트에 필요
TAVILY_API_KEY=your_tavily_api_key_here

# 모델 사용에 필요
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# 선택 사항: 평가 및 추적용
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=deep-agents-from-scratch
```

4.  uv를 사용하여 노트북이나 코드를 실행합니다:

<!-- end list -->

```bash
# Jupyter 노트북 바로 실행
uv run jupyter notebook

# 또는 선호하는 경우 가상 환경 활성화
source .venv/bin/activate  # Windows의 경우: .venv\Scripts\activate
jupyter notebook
```

## 📚 튜토리얼 개요

이 리포지토리에는 고급 AI 에이전트를 구축하는 방법을 알려주는 5개의 점진적인 노트북이 포함되어 있습니다:

### `0_create_agent.ipynb` -

`create_agent` 구성 요소를 사용하는 방법을 배웁니다. 이 구성 요소는,

  - 많은 에이전트의 기반이 되는 ReAct (Reason - Act) 루프를 구현합니다.
  - 사용하기 쉽고 빠르게 설정할 수 있습니다.

### `1_todo.ipynb` - 작업 계획의 기초

TODO 리스트를 사용하여 구조화된 작업 계획을 구현하는 방법을 배웁니다. 이 노트북에서는 다음을 소개합니다:

  - 상태 관리(대기/진행 중/완료)를 통한 작업 추적
  - 진행 상황 모니터링 및 컨텍스트 관리
  - 복잡한 다단계 워크플로우를 구성하기 위한 `write_todos()` 도구
  - 집중을 유지하고 작업 이탈을 방지하기 위한 모범 사례

### `2_files.ipynb` - 가상 파일 시스템

컨텍스트 오프로딩을 위해 에이전트 상태에 저장되는 가상 파일 시스템을 구현합니다:

  - 파일 작업: `ls()`, `read_file()`, `write_file()`, `edit_file()`
  - 정보 지속성을 통한 컨텍스트 관리
  - 대화 턴에 걸쳐 에이전트 "메모리" 활성화
  - 상세 정보를 파일에 저장하여 토큰 사용량 줄이기

### `3_subagents.ipynb` - 컨텍스트 격리

복잡한 워크플로우를 처리하기 위한 하위 에이전트 위임을 마스터합니다:

  - 집중된 도구 세트를 갖춘 특화된 하위 에이전트 생성
  - 혼란 및 작업 간섭을 방지하기 위한 컨텍스트 격리
  - `task()` 위임 도구 및 에이전트 레지스트리 패턴
  - 독립적인 리서치 스트림을 위한 병렬 실행 기능

### `4_full_agent.ipynb` - 완전한 리서치 에이전트

모든 기술을 결합하여 프로덕션 준비가 된 리서치 에이전트를 만듭니다:

  - TODO, 파일, 하위 에이전트의 통합
  - 지능적인 컨텍스트 오프로딩을 사용한 실제 웹 검색
  - 콘텐츠 요약 및 전략적 사고 도구
  - LangGraph Studio와 통합된 복잡한 리서치 작업을 위한 완전한 워크플로우

각 노트북은 이전 개념을 기반으로 구축되며, 실제 세계의 리서치 및 분석 작업을 처리할 수 있는 정교한 에이전트 아키텍처로 마무리됩니다.\`\`\`
