-- DATABASE SETUP
CREATE DATABASE IF NOT EXISTS text_morph;
USE text_morph;

CREATE TABLE IF NOT EXISTS users (
    user_id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(150) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    age INT,
    dob DATE,
    gender ENUM('Male', 'Female', 'Other'),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


CREATE TABLE IF NOT EXISTS otp_verification (
    otp_id INT AUTO_INCREMENT PRIMARY KEY,
    email VARCHAR(150) NOT NULL,
    otp_code VARCHAR(10) NOT NULL,
    purpose ENUM('signup', 'forgot_password') NOT NULL,
    is_verified BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP NOT NULL
);

-- FILE UPLOADS
CREATE TABLE IF NOT EXISTS uploaded_files (
    file_id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    file_name VARCHAR(255) NOT NULL,
    file_type ENUM('pdf', 'docx', 'txt', 'csv') NOT NULL,
    file_content LONGBLOB,
    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

-- SUMMARIZATION HISTORY
CREATE TABLE IF NOT EXISTS summaries (
    summary_id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    input_text LONGTEXT,
    summary_text LONGTEXT,
    model_name VARCHAR(100),
    size ENUM('Small', 'Medium', 'Large'),
    content_category VARCHAR(50) DEFAULT 'General',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

-- PARAPHRASING HISTORY
CREATE TABLE IF NOT EXISTS paraphrases (
    paraphrase_id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    input_text LONGTEXT,
    paraphrase_text LONGTEXT,
    model_name VARCHAR(100),
    size ENUM('Small', 'Medium', 'Large'),
    complexity ENUM('Simple', 'Standard', 'Creative'),
    content_category VARCHAR(50) DEFAULT 'General',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

-- ROUGE SCORES
CREATE TABLE IF NOT EXISTS rouge_scores (
    id INT AUTO_INCREMENT PRIMARY KEY,
    summary_id INT NOT NULL,
    rouge1_f FLOAT,
    rouge2_f FLOAT,
    rougel_f FLOAT,
    rouge1_p FLOAT,
    rouge1_r FLOAT,
    rouge2_p FLOAT,
    rouge2_r FLOAT,
    rougel_p FLOAT,
    rougel_r FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (summary_id) REFERENCES summaries(summary_id) ON DELETE CASCADE
);

-- USER FEEDBACK
CREATE TABLE IF NOT EXISTS feedback (
    feedback_id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    task_type ENUM('summarization', 'paraphrasing') NOT NULL,
    task_id INT NOT NULL,
    rating ENUM('up', 'down') NOT NULL,
    comment TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

-- ADMINS / MENTORS
CREATE TABLE IF NOT EXISTS admins (
    admin_id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL UNIQUE,
    role ENUM('mentor', 'admin') DEFAULT 'admin',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

-- USAGE STATS (Optional Analytics)
CREATE TABLE IF NOT EXISTS usage_stats (
    stat_id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT,
    action ENUM('summarization', 'paraphrasing', 'upload', 'translation', 'evaluation'),
    details TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE SET NULL
);


-- Insert default admin user
INSERT INTO users (name, email, password_hash, age, dob, gender)
VALUES ('Admin', 'ajithreddychittireddy@gmail.com', 'a', NULL, NULL, 'Male')
ON DUPLICATE KEY UPDATE email=email;

-- Get the user_id of the admin
SET @admin_user_id = (SELECT user_id FROM users WHERE email='ajithreddychittireddy@gmail.com');

-- Insert into admins table
INSERT INTO admins (user_id, role)
VALUES (@admin_user_id, 'admin')
ON DUPLICATE KEY UPDATE user_id=user_id;

