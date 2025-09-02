import smtplib
import random
from email.mime.text import MIMEText

# Generate a 6-digit OTP
def generate_otp():
    return str(random.randint(100000, 999999))

# Send OTP using Gmail SMTP
def send_otp_email(receiver_email, otp):
    sender_email = "ajithreddychittireddy@gmail.com"
    sender_password = "imfc oezy kftn qteb"  # Use Gmail App Password (not your main password)

    subject = "Your OTP Code"
    body = f"Your OTP code is {otp}. It will expire in 5 minutes."

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = receiver_email

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, receiver_email, msg.as_string())
        server.quit()
        print(f"OTP sent to {receiver_email}")
        return True
    except Exception as e:
        print("Error sending email:", e)
        return False
