# app/services/slack_service.py
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class SlackService:
    def __init__(self, bot_token: str):
        self.client = WebClient(token=bot_token)

    def send_message(
        self, channel: str, text: str, blocks: Optional[list] = None
    ) -> bool:
        """Slack 채널에 메시지 전송"""
        try:
            logger.info(f"Attempting to send message to channel: {channel}")

            # DM 채널 ID인 경우 (D로 시작), conversations.open을 사용하여 DM 채널 열기
            if channel.startswith("D"):
                logger.info(f"DM channel ID detected, opening conversation...")
                try:
                    # DM 채널 ID를 사용자 ID로 간주하고 conversations.open 시도
                    # 주의: DM 채널 ID와 사용자 ID는 다를 수 있음
                    open_response = self.client.conversations_open(users=[channel])
                    if open_response["ok"]:
                        actual_channel = open_response["channel"]["id"]
                        logger.info(f"Opened DM channel: {actual_channel}")
                        channel = actual_channel
                    else:
                        logger.error(f"Failed to open DM: {open_response}")
                        return False
                except SlackApiError as open_error:
                    error_msg = open_error.response.get("error", str(open_error))
                    logger.error(f"Error opening DM channel: {error_msg}")
                    # conversations.open 실패 시, 원래 채널 ID로 시도
                    logger.warning(f"Trying to send directly to channel {channel}")

            response = self.client.chat_postMessage(
                channel=channel, text=text, blocks=blocks
            )
            if response["ok"]:
                logger.info(f"Successfully sent message to Slack channel {channel}")
            else:
                logger.error(f"Slack API returned ok=False: {response}")
            return response["ok"]
        except SlackApiError as e:
            error_msg = e.response.get("error", str(e))
            logger.error(f"Slack API error: {error_msg}")
            # 상세한 에러 정보 로깅
            if hasattr(e, "response"):
                logger.error(f"Full error response: {e.response}")
                # DM 관련 에러인 경우 추가 정보
                if error_msg in ["channel_not_found", "not_in_channel"]:
                    logger.error(f"Channel {channel} not accessible.")
                    logger.error(
                        "For DM channels, please use your Slack User ID (starts with 'U') instead of DM channel ID"
                    )
                    logger.error(
                        "You can find your User ID in Slack: Profile > More > Copy member ID"
                    )
            return False
        except Exception as e:
            logger.error(f"Unexpected error sending message to Slack: {e}")
            return False

    def format_gmail_message(self, email_data: Dict) -> Dict:
        """Gmail 메시지를 Slack 포맷으로 변환"""
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"📧 New Email: {email_data.get('subject', 'No Subject')}",
                },
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*From:*\\n{email_data.get('sender', 'Unknown')}",
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Date:*\\n{email_data.get('date', 'Unknown')}",
                    },
                ],
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Message:*\\n{email_data.get('snippet', '')[:500]}",
                },
            },
            {"type": "divider"},
        ]

        return {"text": f"New email from {email_data.get('sender')}", "blocks": blocks}

    def send_gmail_notification(self, channel: str, email_data: Dict) -> bool:
        """Gmail 알림을 Slack으로 전송"""
        formatted = self.format_gmail_message(email_data)
        return self.send_message(
            channel=channel, text=formatted["text"], blocks=formatted["blocks"]
        )
