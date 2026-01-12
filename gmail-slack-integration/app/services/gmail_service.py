# app/services/gmail_service.py
import os
import pickle
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from typing import List, Dict, Optional
import base64
from email.mime.text import MIMEText

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]


class GmailService:
    def __init__(self, credentials_path: str, token_path: str):
        self.credentials_path = credentials_path
        self.token_path = token_path
        self.service = None
        self._authenticate()

    def _authenticate(self):
        """Gmail API 인증"""
        creds = None

        # 저장된 토큰 확인
        if os.path.exists(self.token_path):
            with open(self.token_path, "rb") as token:
                creds = pickle.load(token)

        # 토큰이 없거나 만료된 경우
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_path, SCOPES
                )
                # 웹 애플리케이션 타입을 위한 명시적 redirect_uri 설정
                # run_local_server()는 자동으로 마지막에 /를 추가함
                flow.redirect_uri = "http://localhost:5679/"
                creds = flow.run_local_server(port=5679)

            # 토큰 저장
            with open(self.token_path, "wb") as token:
                pickle.dump(creds, token)

        self.service = build("gmail", "v1", credentials=creds)

    def get_unread_messages(self, max_results: int = 10) -> List[Dict]:
        """읽지 않은 이메일 가져오기"""
        try:
            results = (
                self.service.users()
                .messages()
                .list(userId="me", q="is:unread", maxResults=max_results)
                .execute()
            )

            messages = results.get("messages", [])
            return [self.get_message_detail(msg["id"]) for msg in messages]
        except Exception as e:
            print(f"Error fetching messages: {e}")
            return []

    def get_message_detail(self, message_id: str) -> Dict:
        """이메일 상세 정보 가져오기"""
        try:
            message = (
                self.service.users()
                .messages()
                .get(userId="me", id=message_id, format="full")
                .execute()
            )

            headers = message["payload"]["headers"]
            subject = next(
                (h["value"] for h in headers if h["name"] == "Subject"), "No Subject"
            )
            sender = next(
                (h["value"] for h in headers if h["name"] == "From"), "Unknown"
            )
            date = next((h["value"] for h in headers if h["name"] == "Date"), "")

            # 이메일 본문 추출
            body = self._get_message_body(message["payload"])

            return {
                "id": message_id,
                "subject": subject,
                "sender": sender,
                "date": date,
                "body": body,
                "snippet": message.get("snippet", ""),
            }
        except Exception as e:
            print(f"Error fetching message detail: {e}")
            return {}

    def _get_message_body(self, payload: Dict) -> str:
        """이메일 본문 추출"""
        if "parts" in payload:
            for part in payload["parts"]:
                if part["mimeType"] == "text/plain":
                    data = part["body"].get("data", "")
                    if data:
                        return base64.urlsafe_b64decode(data).decode("utf-8")
        elif "body" in payload:
            data = payload["body"].get("data", "")
            if data:
                return base64.urlsafe_b64decode(data).decode("utf-8")
        return ""

    def mark_as_read(self, message_id: str):
        """이메일을 읽음으로 표시"""
        try:
            self.service.users().messages().modify(
                userId="me", id=message_id, body={"removeLabelIds": ["UNREAD"]}
            ).execute()
        except Exception as e:
            print(f"Error marking message as read: {e}")
