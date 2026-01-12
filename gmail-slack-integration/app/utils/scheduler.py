# app/utils/scheduler.py
from apscheduler.schedulers.background import BackgroundScheduler
from app.services.gmail_service import GmailService
from app.services.slack_service import SlackService
from app.config import get_settings
import logging

logger = logging.getLogger(__name__)


class EmailScheduler:
    def __init__(self):
        self.scheduler = BackgroundScheduler()
        self.settings = get_settings()
        self.gmail_service = None
        self.slack_service = None
        self.processed_ids = set()  # 처리된 이메일 ID 추적

    def initialize_services(self):
        """서비스 초기화"""
        self.gmail_service = GmailService(
            credentials_path=self.settings.gmail_credentials_path,
            token_path=self.settings.gmail_token_path,
        )
        self.slack_service = SlackService(bot_token=self.settings.slack_bot_token)

    def check_and_forward_emails(self):
        """이메일 확인 및 Slack 전송"""
        try:
            logger.info("Checking for new emails...")
            messages = self.gmail_service.get_unread_messages()

            new_messages = [
                msg for msg in messages if msg.get("id") not in self.processed_ids
            ]

            if new_messages:
                logger.info(f"Found {len(new_messages)} new emails")

                for message in new_messages:
                    # Slack으로 전송
                    try:
                        success = self.slack_service.send_gmail_notification(
                            channel=self.settings.slack_channel_id, email_data=message
                        )

                        if success:
                            logger.info(f"Sent email {message['id']} to Slack")
                            self.processed_ids.add(message["id"])

                            # 선택적: 이메일을 읽음으로 표시
                            # self.gmail_service.mark_as_read(message['id'])
                        else:
                            logger.error(
                                f"Failed to send email {message['id']} to Slack"
                            )
                    except Exception as e:
                        logger.error(
                            f"Error sending email {message['id']} to Slack: {e}"
                        )
            else:
                logger.info("No new emails found")

        except Exception as e:
            logger.error(f"Error in email check: {e}")

    def start(self):
        """스케줄러 시작"""
        self.initialize_services()

        # 주기적으로 이메일 확인
        self.scheduler.add_job(
            self.check_and_forward_emails,
            "interval",
            seconds=self.settings.gmail_check_interval,
            id="email_check",
        )

        self.scheduler.start()
        logger.info(
            f"Scheduler started - checking emails every {self.settings.gmail_check_interval} seconds"
        )

    def stop(self):
        """스케줄러 중지"""
        self.scheduler.shutdown()
        logger.info("Scheduler stopped")
