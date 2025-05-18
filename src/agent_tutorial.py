import os
from typing import List, Dict
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time

def create_message(role: str, content: str) -> Dict[str, str]:
    return {"role": role, "content": content}

def create_session_with_retries():
    """Create a requests session with retry strategy"""
    session = requests.Session()
    retries = Retry(
        total=3,  # 최대 3번 재시도
        backoff_factor=1,  # 재시도 간격
        status_forcelist=[500, 502, 503, 504]  # 재시도할 HTTP 상태 코드
    )
    session.mount('http://', HTTPAdapter(max_retries=retries))
    return session

def check_ollama_server():
    """Check if Ollama server is running"""
    try:
        session = create_session_with_retries()
        response = session.get("http://localhost:11434/api/tags")
        return True
    except requests.exceptions.ConnectionError:
        return False

def call_ollama_api(messages: List[Dict[str, str]]) -> str:
    """Call Ollama API with conversation history"""
    # 대화 히스토리를 하나의 문자열로 변환
    conversation = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
    
    try:
        session = create_session_with_retries()
        print("\n응답을 기다리는 중... (무제한 대기)")
        response = session.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "anpigon/eeve-korean-10.8b",
                "prompt": conversation,
                "stream": False
            }
        )
        response.raise_for_status()
        return response.json()["response"]
    except requests.exceptions.RequestException as e:
        raise Exception(f"Ollama 서버 통신 오류: {str(e)}")

def chat_loop():
    messages = []
    print("대화를 시작합니다. 종료하려면 '그만'을 입력하세요.")
    
    while True:
        try:
            # 사용자 입력 받기
            user_input = input("\n사용자: ").strip()
            if not user_input:
                continue
                
            if user_input.lower() == '그만':
                print("대화를 종료합니다.")
                break
                
            # 사용자 메시지를 히스토리에 추가
            messages.append(create_message("user", user_input))
            
            # 전체 대화 히스토리를 기반으로 응답 생성
            response_text = call_ollama_api(messages)
            print(f"\n어시스턴트: {response_text}")
            
            # 어시스턴트 응답을 히스토리에 추가
            messages.append(create_message("assistant", response_text))
            
        except Exception as e:
            print(f"\n오류가 발생했습니다: {str(e)}")
            print("다시 시도해주세요.")
            continue  # 오류 발생 시 다음 입력으로 넘어감

def main():
    print("한국어 AI 어시스턴트를 시작합니다...")
    
    if not check_ollama_server():
        print("Error: Ollama 서버가 실행되고 있지 않습니다.")
        print("먼저 Ollama 서버를 실행해주세요.")
        return
        
    chat_loop()

if __name__ == "__main__":
    main()