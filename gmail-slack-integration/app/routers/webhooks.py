# app/routers/webhooks.py
from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel
from app.services.gmail_service import GmailService
from app.services.slack_service import SlackService
from app.config import get_settings

router = APIRouter()
settings = get_settings()

class EmailWebhook(BaseModel):
    message_id: str

@router.post("/gmail/webhook")
async def gmail_webhook(
    webhook_data: EmailWebhook,
    background_tasks: BackgroundTasks
):
    """Gmail Pub/Sub 웹훅 처리 (선택사항)"""
    try:
        gmail_service = GmailService(
            credentials_path=settings.gmail_credentials_path,
            token_path=settings.gmail_token_path
        )
        slack_service = SlackService(bot_token=settings.slack_bot_token)
        
        # 백그라운드에서 처리
        background_tasks.add_task(
            process_email_webhook,
            gmail_service,
            slack_service,
            webhook_data.message_id
        )
        
        return {"status": "processing"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def process_email_webhook(
    gmail_service: GmailService,
    slack_service: SlackService,
    message_id: str
):
    """웹훅 이메일 처리"""
    message = gmail_service.get_message_detail(message_id)
    if message:
        slack_service.send_gmail_notification(
            channel=settings.slack_channel_id,
            email_data=message
        )