"""
😎😎 chat_service_t.py 서빙 관련 서비스

PEFT QLoRA 방식으로 대화하고 학습하는 기능 포함.

세션별 히스토리 관리, 요약, 토큰 절약 전략 등.
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from datasets import Dataset
from langchain_classic.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import PGVector
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    TrainingArguments,
)

try:
    from trl import SFTTrainer
except ImportError:
    from trl.trainer.sft_trainer import SFTTrainer

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings


class ChatService:
    """채팅 서비스 - 모델 로딩 및 RAG 체인 관리."""

    def __init__(
        self,
        connection_string: str,
        collection_name: str,
        model_name_or_path: Optional[str] = None,
    ):
        """ChatService 초기화.

        Args:
            connection_string: PostgreSQL 연결 문자열
            collection_name: PGVector 컬렉션 이름
            model_name_or_path: 로컬 모델 경로 (None이면 기본값 사용)
        """
        self.connection_string = connection_string
        self.collection_name = collection_name
        self.model_name_or_path = model_name_or_path

        # 모델 및 체인
        self.openai_embeddings: Optional[OpenAIEmbeddings] = None
        self.local_embeddings: Optional[HuggingFaceEmbeddings] = None
        self.openai_llm: Optional[ChatOpenAI] = None
        self.local_llm: Optional[Any] = None
        self.openai_rag_chain: Optional[Runnable] = None
        self.local_rag_chain: Optional[Runnable] = None
        self.openai_quota_exceeded = False
        self.vector_store: Optional[PGVector] = None

    def initialize_embeddings(self) -> None:
        """Embedding 모델 초기화 - OpenAI와 로컬 모델 모두 초기화."""
        openai_api_key = os.getenv("OPENAI_API_KEY")

        # OpenAI Embedding 초기화
        if openai_api_key and openai_api_key != "your-api-key-here":
            try:
                self.openai_embeddings = OpenAIEmbeddings()
                # 간단한 테스트
                self.openai_embeddings.embed_query("test")
                print("[OK] OpenAI Embedding 모델 초기화 완료")
            except Exception as e:
                error_msg = str(e)
                if (
                    "quota" in error_msg.lower()
                    or "429" in error_msg
                    or "insufficient_quota" in error_msg
                ):
                    self.openai_quota_exceeded = True
                    print(f"[WARNING] OpenAI API 할당량 초과: {error_msg[:100]}...")
                    print("   OpenAI Embedding을 사용할 수 없습니다.")
                    self.openai_embeddings = None
                else:
                    print(
                        f"[WARNING] OpenAI Embedding 초기화 실패: {error_msg[:100]}..."
                    )
                    self.openai_embeddings = None
        else:
            print("[WARNING] OpenAI API 키가 설정되지 않았습니다.")
            self.openai_embeddings = None

        # 로컬 Embedding 초기화
        try:
            embedding_device = os.getenv("EMBEDDING_DEVICE", "cpu")
            self.local_embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                model_kwargs={"device": embedding_device},
            )
            # 간단한 테스트
            self.local_embeddings.embed_query("test")
            print(
                f"[OK] 로컬 Embedding 모델 초기화 완료 (sentence-transformers, device={embedding_device})"
            )
        except Exception as local_error:
            print(
                f"[WARNING] 로컬 Embedding 모델 초기화 실패: {str(local_error)[:100]}..."
            )
            self.local_embeddings = None

        if not self.openai_embeddings and not self.local_embeddings:
            raise RuntimeError(
                "OpenAI와 로컬 Embedding 모델 모두 초기화에 실패했습니다. "
                "OpenAI API 키를 설정하거나 sentence-transformers를 설치해주세요."
            )

    def initialize_llm(self) -> None:
        """LLM 모델 초기화 - OpenAI와 로컬 모델 모두 초기화."""
        openai_api_key = os.getenv("OPENAI_API_KEY")

        # OpenAI LLM 초기화
        if openai_api_key and openai_api_key != "your-api-key-here":
            try:
                self.openai_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
                # 실제 API 호출 테스트 (할당량 확인)
                self.openai_llm.invoke("test")
                print("[OK] OpenAI Chat 모델 초기화 완료")
            except Exception as e:
                error_msg = str(e)
                if (
                    "quota" in error_msg.lower()
                    or "429" in error_msg
                    or "insufficient_quota" in error_msg
                ):
                    self.openai_quota_exceeded = True
                    print(f"[WARNING] OpenAI API 할당량 초과: {error_msg[:100]}...")
                    print("   OpenAI LLM을 사용할 수 없습니다.")
                    self.openai_llm = None
                else:
                    print(
                        f"[WARNING] OpenAI Chat 모델 초기화 실패: {error_msg[:100]}..."
                    )
                    self.openai_llm = None
        else:
            print("[WARNING] OpenAI API 키가 설정되지 않았습니다.")
            self.openai_llm = None

        # 로컬 Midm LLM 초기화
        try:
            from app.model.model_loader import load_midm_model

            # GPU 메모리 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # .env 파일에서 LOCAL_MODEL_DIR 읽기
            local_model_dir = self.model_name_or_path or os.getenv("LOCAL_MODEL_DIR")
            if local_model_dir:
                # 상대 경로를 절대 경로로 변환
                if not Path(local_model_dir).is_absolute():
                    # 프로젝트 루트 기준으로 변환
                    project_root = Path(__file__).parent.parent.parent.parent
                    local_model_dir = str(project_root / local_model_dir)
                print(f"[INFO] 로컬 모델 디렉토리: {local_model_dir}")
                midm_model = load_midm_model(
                    model_path=local_model_dir, register=False, is_default=False
                )
            else:
                midm_model = load_midm_model(register=False, is_default=False)

            self.local_llm = midm_model.get_langchain_model()
            print("[OK] 로컬 Midm LLM 모델 초기화 완료")
        except Exception as local_error:
            error_msg = str(local_error)
            print(f"[WARNING] 로컬 Midm 모델 초기화 실패: {error_msg[:200]}...")
            import traceback

            print(f"[DEBUG] 상세 오류: {traceback.format_exc()[:500]}")
            self.local_llm = None

        if not self.openai_llm and not self.local_llm:
            raise RuntimeError(
                "OpenAI와 로컬 LLM 모델 모두 초기화에 실패했습니다. "
                "OpenAI API 키를 설정하거나 Midm 모델을 확인해주세요."
            )

    def create_rag_chain(self, llm_model: Any, embeddings_model: Any) -> Runnable:
        """RAG 체인 생성 - LangChain 체인 기능 활용.

        Args:
            llm_model: LLM 모델
            embeddings_model: Embedding 모델

        Returns:
            RAG 체인
        """
        try:
            # 1. Retriever 생성 (현재 Embedding 모델 사용)
            current_vector_store = PGVector(
                embedding_function=embeddings_model,
                collection_name=self.collection_name,
                connection_string=self.connection_string,
            )
            retriever = current_vector_store.as_retriever(search_kwargs={"k": 3})

            # 2. 대화 기록을 고려한 검색 쿼리 생성 프롬프트
            contextualize_q_system_prompt = (
                "대화 기록과 최신 사용자 질문이 주어졌을 때, "
                "대화 기록의 맥락을 참고하여 독립적으로 이해할 수 있는 질문으로 재구성하세요. "
                "질문에 답하지 말고, 필요시 재구성하고 그렇지 않으면 그대로 반환하세요."
            )
            contextualize_q_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", contextualize_q_system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )

            # 3. 대화 기록을 고려한 Retriever 생성
            history_aware_retriever = create_history_aware_retriever(
                llm_model, retriever, contextualize_q_prompt
            )

            # 4. 질문 답변 프롬프트
            qa_system_prompt = (
                "당신은 LangChain과 PGVector를 사용하는 도움이 되는 AI 어시스턴트입니다. "
                "다음 검색된 컨텍스트 정보를 참고하여 사용자의 질문에 답변해주세요. "
                "컨텍스트에 답변할 수 없는 질문이면, 정중하게 그렇게 말씀해주세요. "
                "답변은 최대 3문장으로 간결하게 작성해주세요.\n\n"
                "컨텍스트:\n{context}"
            )
            qa_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", qa_system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )

            # 5. 문서 결합 체인 생성
            question_answer_chain = create_stuff_documents_chain(llm_model, qa_prompt)

            # 6. 최종 RAG 체인 생성
            rag_chain = create_retrieval_chain(
                history_aware_retriever, question_answer_chain
            )

            return rag_chain
        except Exception as e:
            error_msg = str(e)
            print(f"[ERROR] RAG 체인 생성 실패: {error_msg[:200]}...")
            raise

    def initialize_rag_chain(self) -> None:
        """RAG 체인 초기화 - OpenAI와 로컬 모델용 체인 생성."""
        # OpenAI용 RAG 체인 생성
        if self.openai_llm and self.openai_embeddings:
            try:
                self.openai_rag_chain = self.create_rag_chain(
                    self.openai_llm, self.openai_embeddings
                )
                print("[OK] OpenAI RAG 체인 초기화 완료")
            except Exception as e:
                print(f"[WARNING] OpenAI RAG 체인 초기화 실패: {str(e)[:100]}...")
                self.openai_rag_chain = None

        # 로컬 모델용 RAG 체인 생성
        if self.local_llm and self.local_embeddings:
            try:
                self.local_rag_chain = self.create_rag_chain(
                    self.local_llm, self.local_embeddings
                )
                print("[OK] 로컬 RAG 체인 초기화 완료")
            except Exception as e:
                print(f"[WARNING] 로컬 RAG 체인 초기화 실패: {str(e)[:100]}...")
                self.local_rag_chain = None

        if not self.openai_rag_chain and not self.local_rag_chain:
            error_msg = "OpenAI와 로컬 RAG 체인 모두 초기화에 실패했습니다.\n"
            error_msg += "최소 하나의 LLM과 Embedding 모델이 필요합니다."
            print(f"[ERROR] {error_msg}")
            raise RuntimeError(error_msg)

    def chat_with_rag(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
        model_type: str = "openai",
    ) -> str:
        """RAG 체인을 사용하여 대화 생성.

        Args:
            message: 사용자 메시지
            history: 대화 기록
            model_type: 모델 타입 ("openai" 또는 "local")

        Returns:
            생성된 응답
        """
        # 모델 타입 정규화
        if model_type:
            model_type = model_type.lower()
        if model_type == "midm":
            model_type = "local"

        # 적절한 RAG 체인 선택
        if model_type == "openai":
            if not self.openai_rag_chain:
                if self.openai_quota_exceeded:
                    raise RuntimeError("OpenAI API 할당량이 초과되었습니다.")
                else:
                    raise RuntimeError("OpenAI RAG 체인이 초기화되지 않았습니다.")
            current_rag_chain = self.openai_rag_chain
        elif model_type == "local":
            if not self.local_rag_chain:
                raise RuntimeError("로컬 RAG 체인이 초기화되지 않았습니다.")
            current_rag_chain = self.local_rag_chain
        else:
            raise ValueError(f"지원하지 않는 모델 타입입니다: {model_type}")

        # 대화 기록을 LangChain 메시지 형식으로 변환
        chat_history = []
        if history:
            for msg in history:
                if msg.get("role") == "user":
                    chat_history.append(HumanMessage(content=msg.get("content", "")))
                elif msg.get("role") == "assistant":
                    chat_history.append(AIMessage(content=msg.get("content", "")))

        # RAG 체인 실행
        result = current_rag_chain.invoke(
            {
                "input": message,
                "chat_history": chat_history,
            }
        )

        # 체인 결과에서 답변 추출
        response_text = result.get("answer", "답변을 생성할 수 없습니다.")

        # response_text가 None이거나 문자열이 아닌 경우 처리
        if response_text is None:
            response_text = "답변을 생성할 수 없습니다."
        else:
            response_text = str(response_text)

        # 응답에서 이전 대화 내용 제거 (중복 방지)
        if response_text and (
            "Human:" in response_text or "Assistant:" in response_text
        ):
            # 빠른 정규식으로 마지막 Assistant: 이후만 추출
            assistant_match = re.search(
                r"Assistant:\s*(.+?)(?:\nHuman:|$)", response_text, re.DOTALL
            )
            if assistant_match:
                response_text = assistant_match.group(1).strip()

        # 빈 응답 방지
        if not response_text or not response_text.strip():
            response_text = "답변을 생성할 수 없습니다."

        return response_text


class ChatServiceQLoRA:
    """QLoRA를 사용한 채팅 및 학습 서비스."""

    def __init__(
        self,
        model_name_or_path: str,
        output_dir: str = "./qlora_output",
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        target_modules: Optional[List[str]] = None,
        device_map: str = "auto",
    ):
        """QLoRA 채팅 서비스 초기화.

        Args:
            model_name_or_path: 모델 이름 또는 경로
            output_dir: 학습 결과 저장 디렉토리
            lora_r: LoRA rank
            lora_alpha: LoRA alpha
            lora_dropout: LoRA dropout
            target_modules: LoRA를 적용할 모듈 목록 (None이면 자동 감지)
            device_map: 디바이스 매핑 ("auto", "cpu", "cuda" 등)
        """
        self.model_name_or_path = model_name_or_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # QLoRA 설정 (4-bit quantization)
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        # LoRA 설정
        if target_modules is None:
            # 일반적인 모델의 attention 모듈 (Llama, Mistral 등)
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

        self.lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        # 모델 및 토크나이저 로드
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[Any] = None
        self.peft_model: Optional[PeftModel] = None
        self.device_map = device_map

        # 세션별 대화 히스토리
        self.chat_sessions: Dict[str, List[Dict[str, str]]] = {}

    def load_model(self) -> None:
        """모델 및 토크나이저 로드."""
        print(f"[INFO] 모델 로딩 중: {self.model_name_or_path}")

        # 토크나이저 로드
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True,
        )

        # pad_token 설정
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        self.tokenizer = tokenizer

        # 모델 로드 (4-bit quantization)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            quantization_config=self.bnb_config,
            device_map=self.device_map,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

        # PEFT 모델 준비
        model = prepare_model_for_kbit_training(model)

        # LoRA 적용
        peft_model = get_peft_model(model, self.lora_config)
        peft_model.print_trainable_parameters()

        self.model = model
        self.peft_model = peft_model

        print("[OK] 모델 로딩 완료")

    def load_peft_model(self, peft_model_path: str) -> None:
        """학습된 PEFT 모델 로드.

        Args:
            peft_model_path: PEFT 모델 경로
        """
        if self.model is None:
            raise RuntimeError("먼저 load_model()을 호출하세요.")

        print(f"[INFO] PEFT 모델 로딩 중: {peft_model_path}")
        self.peft_model = PeftModel.from_pretrained(
            self.model, peft_model_path, device_map=self.device_map
        )
        print("[OK] PEFT 모델 로딩 완료")

    def chat(
        self,
        message: str,
        session_id: str = "default",
        history: Optional[List[Dict[str, str]]] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """대화 생성.

        Args:
            message: 사용자 메시지
            session_id: 세션 ID
            history: 대화 기록 (None이면 세션 히스토리 사용)
            max_new_tokens: 최대 생성 토큰 수
            temperature: 생성 온도
            top_p: nucleus sampling 파라미터

        Returns:
            생성된 응답
        """
        if self.peft_model is None:
            raise RuntimeError("먼저 load_model() 또는 load_peft_model()을 호출하세요.")

        # 세션 히스토리 가져오기
        if history is None:
            history = self.chat_sessions.get(session_id, [])

        # 대화 형식으로 프롬프트 구성
        prompt = self._format_chat_prompt(message, history)

        # 토크나이징
        if self.tokenizer is None:
            raise RuntimeError("토크나이저가 초기화되지 않았습니다.")

        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=2048
        ).to(self.peft_model.device)

        # 생성
        with torch.no_grad():
            outputs = self.peft_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
                if self.tokenizer.pad_token_id
                else self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # 디코딩
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 응답만 추출 (프롬프트 제외)
        response = generated_text[len(prompt) :].strip()

        # 히스토리 업데이트
        self.chat_sessions[session_id] = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": response},
        ]

        return response

    def _format_chat_prompt(self, message: str, history: List[Dict[str, str]]) -> str:
        """대화 형식으로 프롬프트 구성.

        Args:
            message: 현재 메시지
            history: 대화 기록

        Returns:
            포맷된 프롬프트
        """
        prompt_parts = []

        # 시스템 프롬프트
        prompt_parts.append("당신은 도움이 되는 AI 어시스턴트입니다.")

        # 히스토리 추가
        for msg in history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                prompt_parts.append(f"사용자: {content}")
            elif role == "assistant":
                prompt_parts.append(f"어시스턴트: {content}")

        # 현재 메시지 추가
        prompt_parts.append(f"사용자: {message}")
        prompt_parts.append("어시스턴트:")

        return "\n".join(prompt_parts)

    def train(
        self,
        training_data: List[Dict[str, str]],
        output_dir: Optional[str] = None,
        num_epochs: int = 3,
        per_device_train_batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        learning_rate: float = 2e-4,
        warmup_steps: int = 100,
        logging_steps: int = 10,
        save_steps: int = 500,
        max_seq_length: int = 512,
    ) -> str:
        """QLoRA 학습 실행.

        Args:
            training_data: 학습 데이터 ({"instruction": "...", "input": "...", "output": "..."} 형식)
            output_dir: 출력 디렉토리 (None이면 기본값 사용)
            num_epochs: 에폭 수
            per_device_train_batch_size: 배치 크기
            gradient_accumulation_steps: 그래디언트 누적 스텝
            learning_rate: 학습률
            warmup_steps: 워밍업 스텝
            logging_steps: 로깅 스텝
            save_steps: 저장 스텝
            max_seq_length: 최대 시퀀스 길이

        Returns:
            학습된 모델 경로
        """
        if self.peft_model is None:
            raise RuntimeError("먼저 load_model()을 호출하세요.")

        if self.tokenizer is None:
            raise RuntimeError("토크나이저가 초기화되지 않았습니다.")

        output_dir = output_dir or str(
            self.output_dir / f"checkpoint-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )

        # 데이터셋 준비
        def format_prompt(example):
            """프롬프트 포맷팅."""
            instruction = example.get("instruction", "")
            input_text = example.get("input", "")
            output = example.get("output", "")

            if input_text:
                prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
            else:
                prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"

            return {"text": prompt}

        dataset = Dataset.from_list(training_data)
        dataset = dataset.map(format_prompt)

        # 학습 인자 설정
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            logging_steps=logging_steps,
            save_steps=save_steps,
            save_total_limit=3,
            fp16=False,  # QLoRA는 bfloat16 사용
            bf16=True,
            optim="paged_adamw_8bit",
            lr_scheduler_type="cosine",
            report_to="none",
        )

        # 데이터 콜레이터
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False
        )

        # 트레이너 생성
        trainer_kwargs: Dict[str, Any] = {
            "model": self.peft_model,
            "train_dataset": dataset,
            "peft_config": self.lora_config,
            "tokenizer": self.tokenizer,
            "args": training_args,
            "data_collator": data_collator,
            "max_seq_length": max_seq_length,
        }

        # packing 파라미터는 버전에 따라 선택적
        try:
            trainer = SFTTrainer(**trainer_kwargs, packing=False)  # type: ignore
        except TypeError:
            # packing 파라미터가 없는 경우
            trainer_kwargs.pop("packing", None)
            trainer = SFTTrainer(**trainer_kwargs)  # type: ignore

        # 학습 실행
        print("[INFO] 학습 시작...")
        trainer.train()
        print("[OK] 학습 완료")

        # 모델 저장
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)

        print(f"[OK] 모델 저장 완료: {output_dir}")
        return output_dir

    def train_from_chat_history(
        self,
        session_ids: Optional[List[str]] = None,
        output_dir: Optional[str] = None,
        **train_kwargs,
    ) -> str:
        """채팅 히스토리로부터 학습 데이터 생성 및 학습.

        Args:
            session_ids: 학습할 세션 ID 목록 (None이면 모든 세션)
            output_dir: 출력 디렉토리
            **train_kwargs: train() 메서드에 전달할 추가 인자

        Returns:
            학습된 모델 경로
        """
        # 학습 데이터 생성
        training_data = []

        if session_ids is None:
            session_ids = list(self.chat_sessions.keys())

        for session_id in session_ids:
            history = self.chat_sessions.get(session_id, [])
            if len(history) < 2:
                continue

            # 대화 쌍으로 변환
            for i in range(0, len(history) - 1, 2):
                if i + 1 < len(history):
                    user_msg = history[i].get("content", "")
                    assistant_msg = history[i + 1].get("content", "")

                    training_data.append(
                        {
                            "instruction": "다음 대화에 응답하세요.",
                            "input": user_msg,
                            "output": assistant_msg,
                        }
                    )

        if not training_data:
            raise ValueError("학습할 데이터가 없습니다.")

        print(f"[INFO] {len(training_data)}개의 학습 샘플 생성됨")

        # 학습 실행
        return self.train(training_data, output_dir=output_dir, **train_kwargs)

    def save_session(self, session_id: str, file_path: str) -> None:
        """세션 히스토리 저장.

        Args:
            session_id: 세션 ID
            file_path: 저장 경로
        """
        history = self.chat_sessions.get(session_id, [])
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        print(f"[OK] 세션 저장 완료: {file_path}")

    def load_session(self, session_id: str, file_path: str) -> None:
        """세션 히스토리 로드.

        Args:
            session_id: 세션 ID
            file_path: 로드 경로
        """
        with open(file_path, "r", encoding="utf-8") as f:
            history = json.load(f)
        self.chat_sessions[session_id] = history
        print(f"[OK] 세션 로드 완료: {file_path}")

    def clear_session(self, session_id: str) -> None:
        """세션 히스토리 삭제.

        Args:
            session_id: 세션 ID
        """
        if session_id in self.chat_sessions:
            del self.chat_sessions[session_id]
            print(f"[OK] 세션 삭제 완료: {session_id}")

    def get_session_history(self, session_id: str) -> List[Dict[str, str]]:
        """세션 히스토리 가져오기.

        Args:
            session_id: 세션 ID

        Returns:
            대화 기록 리스트
        """
        return self.chat_sessions.get(session_id, [])
