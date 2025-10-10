from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

def build_chain(llm_choice, retriever, google_key, nvidia_key):
    #llm---------------------------------------------------------------------------------
    if llm_choice == "gemini":
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=google_key)
    elif llm_choice == "nvidia":
        llm = ChatNVIDIA(temperature=0, nvidia_api_key=nvidia_key)
    else:
        raise ValueError("Invalid LLM choice")

    #prompt---------------------------------------------------------------------------------
    prompt = PromptTemplate(
        template="""You are a helpful assistant.
        Answer ONLY from the provided transcript context.
        If the context is insufficient, just say you don't know.

        {context}
        Question: {question}""",
        input_variables=["context", "question"],
    )

    #context---------------------------------------------------------------------------------
    def generate_context(retrieved_docs):
        context="\n".join([doc.page_content for doc in retrieved_docs])
        return context
    


    #final prompt---------------------------------------------------------------------------------
    def generate_prompt(retriever, question):
        prompt_chain=RunnableParallel(
            {
            'context': retriever | RunnableLambda(generate_context),
            'question': RunnablePassthrough()
        }
        )
        prompt=prompt_chain.invoke(question)
        return prompt
    
    #final prompt chain---------------------------------------------------------------------------------
    generate_prompt_chain_f=RunnableLambda(lambda question: generate_prompt(retriever, question))

    #parser---------------------------------------------------------------------------------
    output_parser = StrOutputParser()

    #main chain ---------------------------------------------------------------------------------
    chain= generate_prompt_chain_f|prompt|llm|output_parser

    return chain



#RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type_kwargs={"prompt": prompt})