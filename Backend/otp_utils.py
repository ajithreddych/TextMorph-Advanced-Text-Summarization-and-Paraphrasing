import smtplib
import random
from email.mime.text import MIMEText

def generate_otp(length=6):
    return str(random.randint(10**(length-1), 10**length-1))

def send_otp_email(receiver_email, otp_code):
    sender_email = "ajithreddychittireddy3@gmail.com"
    sender_password = "cqdm fhow haic xyyj"  # App password

    subject = "Your OTP for Text Morph – Advanced Summarisation using AI"
    body = f"Your OTP code is: {otp_code}"

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = receiver_email

    try:
        print(f"📧 Sending OTP to {receiver_email}...")
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, receiver_email, msg.as_string())
        print("✅ OTP sent successfully!")
        return True
    except Exception as e:
        print(f"❌ Failed to send OTP: {e}")
        return False
