from flask import Flask
import account
from user_profile import profile_bp # import the routes; they already use app = Flask(__name__)
from summarization_routes import summarization_bp
from translation_routes import translate_bp
from feedback_routes import feedback_bp
from paraphrase_routes import paraphrase_bp 
from history import history_bp
from dataset import dataset_bp
from admin_view import admin_bp
from admin_view_usage import admin_usage_bp


app = account.app
app.register_blueprint(profile_bp)
app.register_blueprint(summarization_bp, url_prefix="/api")
app.register_blueprint(translate_bp)
app.register_blueprint(feedback_bp, url_prefix="/feedback")
app.register_blueprint(paraphrase_bp, url_prefix="/api")
app.register_blueprint(history_bp, url_prefix="/history")
app.register_blueprint(dataset_bp, url_prefix="/dataset") 
app.register_blueprint(admin_bp, url_prefix="/admin")
app.register_blueprint(admin_usage_bp, url_prefix="/admin")
 # use the app defined in account.py

if __name__ == "__main__":
    app.run(debug=True)
