from flask_mail import Message, Mail
from app.config.config import DevelopmentConfig

mail = Mail()


def send_email(to, subject, template):
    msg = Message(
        subject,
        recipients=[to],
        html=template,
        sender=DevelopmentConfig.MAIL_DEFAULT_SENDER
    )

    mail.send(msg)
