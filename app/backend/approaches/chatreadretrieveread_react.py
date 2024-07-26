from typing import Any, Coroutine, List, Literal, Optional, Union, overload

from azure.search.documents.aio import SearchClient
from azure.search.documents.models import VectorQuery
from openai import AsyncOpenAI, AsyncStream, AzureOpenAI, AsyncAzureOpenAI
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)
from openai_messages_token_helper import build_messages, get_token_limit

from approaches.approach import ThoughtStep, Document, QueryCaptionResult, QueryType
from approaches.chatapproach import ChatApproach
from core.authentication import AuthenticationHelper
from collections.abc import Callable
import os
from azure.identity.aio import DefaultAzureCredential, get_bearer_token_provider
import json
from typing import (
    Any,
    AsyncGenerator,
    Awaitable,
    Callable,
    List,
    Optional,
    TypedDict,
    cast,
)


class ChatReadRetrieveReadReactApproach(ChatApproach):
    """
    A multi-step approach that first uses OpenAI to turn the user's question into a search query,
    then uses Azure AI Search to retrieve relevant documents, and then sends the conversation history,
    original user question, and search results to OpenAI to generate a response.
    """

    def __init__(
        self,
        *,
        search_client: SearchClient,
        auth_helper: AuthenticationHelper,
        openai_client: AsyncOpenAI,
        chatgpt_model: str,
        chatgpt_deployment: Optional[str],  # Not needed for non-Azure OpenAI
        embedding_deployment: Optional[str],  # Not needed for non-Azure OpenAI or for retrieval_mode="text"
        embedding_model: str,
        embedding_dimensions: int,
        sourcepage_field: str,
        content_field: str,
        query_language: str,
        query_speller: str,
    ):
        self.search_client = search_client
        self.openai_client = openai_client
        self.auth_helper = auth_helper
        self.chatgpt_model = chatgpt_model
        self.chatgpt_deployment = chatgpt_deployment
        self.embedding_deployment = embedding_deployment
        self.embedding_model = embedding_model
        self.embedding_dimensions = embedding_dimensions
        self.sourcepage_field = sourcepage_field
        self.content_field = content_field
        self.query_language = query_language
        self.query_speller = query_speller
        self.chatgpt_token_limit = get_token_limit(chatgpt_model)

    @property
    def system_message_chat_conversation(self):
        return """Assistant helps the business analysts with their questions about the swiss based insurance company Baloise and Helvetia, specifically about their annual reports for 2023 or general insurance conditions on motercycles. Be brief in your answers.
        Answer ONLY with the facts listed in the list of sources below. If there isn't enough information below, say you don't know. Do not generate answers that don't use the sources below. If asking a clarifying question to the user would help, ask the question.
        For tabular information return it as an html table. Do not return markdown format. If the question is not in English, answer in the language used in the question.
        Each source has a name followed by colon and the actual information, always include the source name for each fact you use in the response. Use square brackets to reference the source, for example [info1.txt]. Don't combine sources, list each source separately, for example [info1.txt][info2.pdf].
        You have access to a set of tools/functions that helps the user searches on the web, or generates marketing images, or search internal stock videos, or searches internal knowledge base on Nescafé's brand guidelines.
        IMPORTANT: It is crucial to contextualize first what is the user request really about based on user intent and chat history as your context, and then choose the function to use. Slow down and think step by step.
        You can answer complex multistep questions by sequentially or parallelly calling functions. Follow a pattern of 
            THOUGHT (reason step-by-step about which function to call next), 
            ACTION (call a function to as a next step towards the final answer), 
            OBSERVATION (output of the function). Reason step by step which actions to take to get to the answer. 
        Only call functions with arguments coming verbatim from the user or the output of other functions. In your final answer to the user, always include the source page for each fact you use in the response.
        {follow_up_questions_prompt}
        {injected_prompt}
        """

    @overload
    async def run_until_final_call(
        self,
        messages: list[ChatCompletionMessageParam],
        overrides: dict[str, Any],
        auth_claims: dict[str, Any],
        should_stream: Literal[False],
    ) -> tuple[dict[str, Any], Coroutine[Any, Any, ChatCompletion]]: ...

    @overload
    async def run_until_final_call(
        self,
        messages: list[ChatCompletionMessageParam],
        overrides: dict[str, Any],
        auth_claims: dict[str, Any],
        should_stream: Literal[True],
    ) -> tuple[dict[str, Any], Coroutine[Any, Any, AsyncStream[ChatCompletionChunk]]]: ...
    
    async def doc_search(
        self,
        top: Optional[int] = 5,
        query_text: Optional[str] = None,
        filter: Optional[str] = None,
        use_text_search: bool = True,
        use_vector_search: bool = True,
        use_semantic_ranker: bool = True,
        use_semantic_captions: bool = False,
        minimum_search_score: Optional[float] = None,
        minimum_reranker_score: Optional[float] = None,
    ) -> List[Document]:
        search_text = query_text if use_text_search else ""
        vectors: list[VectorQuery] = []
        if use_vector_search:
            vectors.append(await self.compute_text_embedding(query_text))
        search_vectors = vectors if use_vector_search else []
        if use_semantic_ranker:
            results = await self.search_client.search(
                search_text=search_text,
                filter=filter,
                top=top,
                query_caption="extractive|highlight-false" if use_semantic_captions else None,
                vector_queries=search_vectors,
                query_type=QueryType.SEMANTIC,
                query_language=self.query_language,
                query_speller=self.query_speller,
                semantic_configuration_name="default",
                semantic_query=query_text,
            )
        else:
            results = await self.search_client.search(
                search_text=search_text,
                filter=filter,
                top=top,
                vector_queries=search_vectors,
            )

        documents = []
        async for page in results.by_page():
            async for document in page:
                documents.append(
                    Document(
                        id=document.get("id"),
                        content=document.get("content"),
                        embedding=document.get("embedding"),
                        image_embedding=document.get("imageEmbedding"),
                        category=document.get("category"),
                        sourcepage=document.get("sourcepage"),
                        sourcefile=document.get("sourcefile"),
                        oids=document.get("oids"),
                        groups=document.get("groups"),
                        captions=cast(List[QueryCaptionResult], document.get("@search.captions")),
                        score=document.get("@search.score"),
                        reranker_score=document.get("@search.reranker_score"),
                    )
                )

            qualified_documents = [
                doc
                for doc in documents
                if (
                    (doc.score or 0) >= (minimum_search_score or 0)
                    and (doc.reranker_score or 0) >= (minimum_reranker_score or 0)
                )
            ]

            sources_content = self.get_sources_content(qualified_documents, use_semantic_captions, use_image_citation=False)
            content = "\n\nSources:\n" + "\n".join(sources_content)

        return content
    
    def finish(self, answer) -> None:
        return answer

    async def run_until_final_call(
        self,
        messages: list[ChatCompletionMessageParam],
        overrides: dict[str, Any],
        auth_claims: dict[str, Any],
        should_stream: bool = False,
    ) -> tuple[dict[str, Any], Coroutine[Any, Any, Union[ChatCompletion, AsyncStream[ChatCompletionChunk]]]]:

        original_user_query = messages[-1]["content"]
        if not isinstance(original_user_query, str):
            raise ValueError("The most recent message content must be a string.")

        tools: List[ChatCompletionToolParam] = [
            {
                "type": "function",
                "function": {
                    "name": "doc_search",
                    "description": "use this function to retrieve sources from the Azure AI Search index to answer any questions around Baloise and Helvetia.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query_text": {
                                "type": "string",
                                "description": "the rephrased user request in one concise sentence considering all the important context from the conversation history",
                            }
                        },
                        "required": ["query_text"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "finish",
                    "description": "use this function when the current answer is already sufficient to the user's question, and finish the conversation.",
                    "parameters": {
                    "type": "object",
                    "properties": {
                        "answer": {
                            "type": "string",
                            "description": "Answer to the user's question."
                        }
                    },
                    "required": [
                        "answer"
                    ]
                    }
                },
            },
        ]
        # All functions that can be called by the LLM Agent
        name_to_function_map: dict[str, Callable] = {
            self.doc_search.__name__: self.doc_search,
            self.finish.__name__: self.finish,
        }
        async def run(messages: list[dict]):
            """
            Run the ReAct loop with OpenAI Function Calling.
            """
            final_output = None
            AZURE_OPENAI_SERVICE = os.getenv("AZURE_OPENAI_SERVICE")
            endpoint = f"https://{AZURE_OPENAI_SERVICE}.openai.azure.com"
            api_version = "2024-02-01" # 2024-02-15-preview, 2023-12-01-preview
            model = os.getenv("AZURE_OPENAI_CHATGPT_DEPLOYMENT")
            azure_credential = DefaultAzureCredential(exclude_shared_token_cache_credential=True)
            token_provider = get_bearer_token_provider(azure_credential, "https://cognitiveservices.azure.com/.default")
            openai_client = AsyncAzureOpenAI(
                api_version=api_version,
                azure_endpoint=endpoint,
                azure_ad_token_provider=token_provider,
            )

            system_message = self.get_system_prompt(
                overrides.get("prompt_template"),
                self.follow_up_questions_prompt_content if overrides.get("suggest_followup_questions") else "",
            )
            response_token_limit = 2048
            messages = build_messages(
                model=self.chatgpt_model,
                system_prompt=system_message,
                past_messages=messages[:-1],
                # Model does not handle lengthy system messages well. Moving sources to latest user conversation to solve follow up questions prompt.
                new_user_content= messages[-1]["content"],
                max_tokens=self.chatgpt_token_limit - response_token_limit,
            )
            internal_log = ""
            # Run in loop
            max_iterations = 10
            for i in range(max_iterations): 
                internal_log += f"Iteration {i}: Preparing to generate response.\n"
                print(f"Iteration {i}: Preparing to generate response.") 

                # Send list of messages to get next response
                response = await openai_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                )
                response_message = response.choices[0].message
                messages.append(response_message)  # Extend conversation with assistant's reply
                internal_log += f"Iteration {i}: Response generated: {response_message.content}\n"
                print(f"Iteration {i}: Response generated: {response_message.content}")
                # print(f"added message: {response_message}")
                tool_calls = response_message.tool_calls
                # Check if GPT wanted to call a function
                if tool_calls:
                    for tool_call in tool_calls:
                        function_name = tool_call.function.name
                        internal_log += f"Iteration {i}: Function {function_name} is selected\n"
                        print(f"Iteration {i}: Function {function_name} is selected")
                        # Validate function name
                        if function_name not in name_to_function_map:
                            internal_log += f"Invalid function name: {function_name}\n"
                            print(f"Invalid function name: {function_name}")
                            messages.append(
                                {
                                    "tool_call_id": tool_call.id,
                                    "role": "tool",
                                    "name": function_name,
                                    "content": f"Invalid function name: {function_name!r}",
                                }
                            )
                            continue
                        # Get the function to call
                        function_to_call: Callable = name_to_function_map[function_name]
                        # Try getting the function arguments
                        try:
                            function_args_dict = json.loads(tool_call.function.arguments)
                        except json.JSONDecodeError as exc:
                            # JSON decoding failed
                            print(f"Error decoding function arguments: {exc}")
                            messages.append(
                                {
                                    "tool_call_id": tool_call.id,
                                    "role": "tool",
                                    "name": function_name,
                                    "content": f"Error decoding function call `{function_name}` arguments {tool_call.function.arguments!r}! Error: {exc!s}",
                                }
                            )
                            continue
                        # Call the selected function with generated arguments
                        internal_log += f"Iteration {i}: Calling function {function_name} with args: {json.dumps(function_args_dict)}\n"
                        print(f"Iteration {i}: Calling function {function_name} with args: {json.dumps(function_args_dict)}")  
            
                        # ... 
                        if function_name == "finish":
                            internal_log += f"Iteration {i}: Function calling is finished, outputing to frontend.\n"
                            final_output = function_args_dict["answer"]
                            final_output = {
                            "choices": [
                                {
                                    "message": {
                                        "content": final_output
                                    }
                                }
                            ]
                        }
                            return final_output, internal_log
                        else:
                            function_response = await function_to_call(**function_args_dict)
                            internal_log += f"Iteration {i}: Function {function_name} returned response: {function_response}\n"
                            print(f"Function {function_name} returned response: {function_response}")
                            # Extend conversation with function response
                            messages.append(
                                {
                                    "tool_call_id": tool_call.id,
                                    "role": "tool",
                                    "name": function_name,
                                    "content": json.dumps(function_response, ensure_ascii=False),
                                }
                            )
                else: 
                    internal_log += f"Iteration {i}: No function call detected, outputing to frontend.\n"
                    print(f"Iteration {i}: No function call detected, outputing to frontend.")  
                    final_output = response_message.content
                    final_output = {
                            "choices": [
                                {
                                    "message": {
                                        "content": final_output
                                    }
                                }
                            ]
                        }
                
            return final_output, internal_log
        
        final_output, internal_log = await run(messages)
        print("Final output: ", final_output)
        print("Internal log: ", internal_log)
        chat_coroutine = final_output
        data_points = {"text": internal_log}

        extra_info = {
            "data_points": data_points,
            "thoughts": []
            }
        # Return the output as a JSON response  
        return (extra_info, chat_coroutine)