from fastapi import APIRouter

mail_router = APIRouter(prefix="/mail", tags=["mail"])

@mail_router.post("/")
async def send_mail(mail: EmailRequest):
    pass

@mail_router.post("/")
async def spam_mail_filter(mail: EmailRequest):
    pass

