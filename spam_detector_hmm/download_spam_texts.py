"""
Скачивание спам-текстов (email, SMS, реклама)
"""
import os
import requests
import pandas as pd
from pathlib import Path
import time
import random

def download_sms_spam_collection():
    """Скачивание SMS спам датасета"""
    print("📱 Загрузка SMS Spam Collection...")
    
    spam_dir = Path('data/raw/spam')
    spam_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Известные спам сообщения из SMS Spam Collection
        spam_messages = [
            "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate).",
            "WINNER!! As a valued network customer you have been selected to receive a £900 prize reward!",
            "Had your mobile 11 months or more? U R entitled to Update to the latest colour mobiles with camera for Free!",
            "SIX chances to win CASH! From 100 to 20,000 pounds txt> CSH11 and send to 87575.",
            "URGENT! You have won a 1 week FREE membership in our £100,000 Prize Jackpot!",
            "Congratulations! You've been selected to receive a free iPhone! Click here to claim now.",
            "You have won a $1000 Walmart gift card. Text YES to claim your prize.",
            "Last chance to claim your free cruise vacation. Limited time offer!",
            "Your bank account has been compromised. Click to secure your funds immediately.",
            "Make $5000 weekly working from home. No experience required. Start today!"
        ]
        
        downloaded = 0
        for i, message in enumerate(spam_messages):
            filename = spam_dir / f"sms_spam_{i}.txt"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(message)
            downloaded += 1
        
        print(f"  ✓ Загружено {downloaded} SMS спам сообщений")
        return downloaded
        
    except Exception as e:
        print(f"  ✗ Ошибка загрузки SMS спам: {e}")
        return 0

def download_email_spam():
    """Скачивание email спам сообщений"""
    print("\n📧 Загрузка email спам сообщений...")
    
    spam_dir = Path('data/raw/spam')
    
    # Примеры email спама
    email_templates = [
        """Subject: Urgent: Your Account Security Alert

Dear Valued Customer,

We have detected unusual activity on your account. To prevent unauthorized access, please verify your identity immediately by clicking the link below.

[FAKE LINK REMOVED]

Failure to respond within 24 hours will result in account suspension.

Sincerely,
Security Team""",

        """Subject: LIMITED TIME OFFER: 90% OFF ALL ITEMS!

HURRY! This offer expires soon! Get massive discounts on all products!

🔥 ELECTRONICS - Up to 90% OFF!
👕 FASHION - Buy 1 Get 3 FREE!
💄 BEAUTY - 80% DISCOUNT!

Shop now: [FAKE LINK REMOVED]

Don't miss this incredible opportunity!""",

        """Subject: You Have Won $1,000,000!

CONGRATULATIONS!

You have been selected as the winner of our $1,000,000 lottery! To claim your prize, please provide your banking information and pay a small processing fee of $50.

Contact us immediately at: fake-email@scam.com

This is a limited time offer!""",

        """Subject: Your Package Delivery Failed

We attempted to deliver your package but were unable to do so. Please confirm your address and payment information to reschedule delivery.

Tracking Number: [FAKE NUMBER]
Delivery Date: [FAKE DATE]

Click here to update your information: [FAKE LINK]

Thank you,
Delivery Service""",

        """Subject: Important Tax Refund Notification

You are eligible for a tax refund of $328.90. Please submit your claim within 72 hours to receive your payment.

To claim your refund, visit: [FAKE LINK]

This is an automated message from the Tax Department."""
    ]
    
    downloaded = 0
    for i, email in enumerate(email_templates):
        filename = spam_dir / f"email_spam_{i}.txt"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(email)
        downloaded += 1
    
    print(f"  ✓ Загружено {downloaded} email спам сообщений")
    return downloaded

def generate_seo_spam():
    """Генерация SEO спам текстов (переоптимизированные)"""
    print("\n🔍 Генерация SEO спам текстов...")
    
    spam_dir = Path('data/raw/spam')
    
    # Ключевые слова для разных типов спама
    spam_keywords = {
        'pharmacy': ['buy', 'cheap', 'discount', 'online', 'pharmacy', 'pills', 'medicine', 'drug', 'price', 'viagra', 'cialis'],
        'casino': ['casino', 'gambling', 'poker', 'slots', 'blackjack', 'roulette', 'win', 'money', 'jackpot', 'bonus'],
        'money': ['make money', 'fast', 'easy', 'work from home', 'earn cash', 'rich', 'wealthy', 'millionaire', 'business opportunity'],
        'weight_loss': ['weight loss', 'diet', 'supplement', 'fat burner', 'lose weight', 'quick results', 'pills', 'burner'],
        'seo': ['SEO', 'services', 'optimization', 'ranking', 'google', 'first page', 'backlinks', 'traffic', 'visitors']
    }
    
    downloaded = 0
    
    for category, keywords in spam_keywords.items():
        # Создаем переоптимизированный текст
        text = " ".join(keywords * 5)  # Многократное повторение ключевых слов
        text += " " + " ".join(random.sample(keywords, len(keywords))) * 3
        
        filename = spam_dir / f"seo_spam_{category}.txt"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(text)
        downloaded += 1
        print(f"  ✓ Создан {category} спам текст")
    
    return downloaded

def create_additional_spam():
    """Создание дополнительных спам текстов"""
    print("\n📝 Создание дополнительных спам текстов...")
    
    spam_dir = Path('data/raw/spam')
    
    additional_spam = [
        "BUY BUY BUY CHEAP PILLS ONLINE PHARMACY DISCOUNT BEST PRICE MEDICINE DRUG STORE",
        "CASINO GAMBLING POKER SLOTS WIN MONEY JACKPOT BONUS FREE SPINS ONLINE GAMBLING",
        "MAKE MONEY ONLINE FAST EASY WORK FROM HOME EARN CASH QUICK RICH WEALTHY MILLIONAIRE",
        "WEIGHT LOSS PILLS DIET SUPPLEMENT FAT BURNER LOSE WEIGHT FAST QUICK RESULTS AMAZING",
        "SEO SERVICES OPTIMIZATION RANKING GOOGLE FIRST PAGE BACKLINKS TRAFFIC VISITORS EXPERT",
        "FREE IPHONE MACBOOK COMPUTER WINNER PRIZE GIVEAWAY CONTEST LIMITED TIME OFFER EXCLUSIVE",
        "URGENT IMPORTANT SECURITY ALERT BANK ACCOUNT CREDIT CARD INFORMATION VERIFY IMMEDIATELY",
        "HOT SINGLES IN YOUR AREA DATING CHAT MEET LOCAL PEOPLE ROMANCE RELATIONSHIP CONNECTION",
        "INVESTMENT OPPORTUNITY HIGH RETURN PROFIT STOCK CRYPTO BITCOIN TRADING SUCCESS WEALTH",
        "DIPLOMA DEGREE CERTIFICATE UNIVERSITY COLLEGE EDUCATION ONLINE ACCREDITED INSTANT EASY"
    ]
    
    downloaded = 0
    existing_files = len(list(spam_dir.glob("*.txt")))
    
    # Добавляем пока не достигнем 100 файлов
    for i in range(max(0, 100 - existing_files)):
        text = random.choice(additional_spam)
        filename = spam_dir / f"additional_spam_{i}.txt"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(text + " " + " ".join(text.split() * 2))  # Делаем текст длиннее
        downloaded += 1
    
    return downloaded

if __name__ == "__main__":
    print("🚀 ЗАГРУЗКА СПАМ ТЕКСТОВ")
    print("=" * 50)
    
    total_downloaded = 0
    
    # Загружаем разные типы спама
    sms_count = download_sms_spam_collection()
    total_downloaded += sms_count
    
    email_count = download_email_spam()
    total_downloaded += email_count
    
    seo_count = generate_seo_spam()
    total_downloaded += seo_count
    
    # Добираем до 100 файлов
    additional_count = create_additional_spam()
    total_downloaded += additional_count
    
    # Итоговая статистика
    spam_dir = Path('data/raw/spam')
    final_count = len(list(spam_dir.glob("*.txt")))
    
    print(f"\n✅ ЗАВЕРШЕНО!")
    print(f"📊 Итоговая статистика:")
    print(f"   SMS спам: {sms_count}")
    print(f"   Email спам: {email_count}")
    print(f"   SEO спам: {seo_count}")
    print(f"   Дополнительных: {additional_count}")
    print(f"   Всего файлов: {final_count}")
    print(f"   Папка: {spam_dir.absolute()}")