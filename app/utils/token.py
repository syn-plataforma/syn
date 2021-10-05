import os

from itsdangerous import URLSafeTimedSerializer

from app.config.config import DevelopmentConfig


def generate_verification_token(email):
    serializer = URLSafeTimedSerializer(os.environ.get('SECRET_KEY', DevelopmentConfig.SECRET_KEY))
    return serializer.dumps(email,
                            salt=os.environ.get('SECURITY_PASSWORD_SALT', DevelopmentConfig.SECURITY_PASSWORD_SALT))


def confirm_verification_token(token, expiration=3600):
    serializer = URLSafeTimedSerializer(os.environ.get('SECRET_KEY', DevelopmentConfig.SECRET_KEY))
    try:
        email = serializer.loads(
            token,
            salt=os.environ.get('SECURITY_PASSWORD_SALT', DevelopmentConfig.SECURITY_PASSWORD_SALT),
            max_age=expiration
        )
    except Exception as e:
        return e
    return email
