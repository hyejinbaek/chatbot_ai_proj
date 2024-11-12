# langchain==0.1.16
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryBufferMemory

openai_api_key="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"

# 대화 모델과 메모리 설정
llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key = openai_api_key)
memory = ConversationSummaryBufferMemory(llm=llm)

conversation = ConversationChain(llm=llm, memory=memory)

# 대화 예시
conversation.invoke("무지개 색깔이 뭐죠?")
conversation.invoke("다음 색깔들은 뭐죠?")
# 대화 예시
print(conversation.invoke("무지개 색깔이 뭐죠?"))
print(conversation.invoke("다음 색깔들은 뭐죠?"))
